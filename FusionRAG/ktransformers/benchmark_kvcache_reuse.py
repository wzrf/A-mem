"""
Benchmark script to verify KVCache reuse speedup
Compares full prefill vs cache reuse with random 30% recompute for 16K token text
"""
import torch
import os
import sys
import time
import random
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.models.custom_cache import StaticCache
from ktransformers.models.modeling_openpangu_dense import PanguEmbeddedForCausalLM


def generate_long_text(tokenizer, target_length=16000):
    """
    Generate a long text with approximately target_length tokens
    """
    # Use a repeating pattern to create long text
    base_text = """The rapid advancement of artificial intelligence has transformed numerous industries and aspects of daily life. Machine learning algorithms, particularly deep neural networks, have demonstrated remarkable capabilities in tasks ranging from image recognition to natural language processing. These systems learn patterns from vast amounts of data, enabling them to make predictions and decisions with increasing accuracy.

In recent years, large language models have emerged as a breakthrough technology. These models, trained on billions of words from diverse sources, can generate coherent text, answer questions, and even engage in creative writing. The transformer architecture, introduced in 2017, revolutionized the field by enabling parallel processing of sequences and capturing long-range dependencies in text.

However, the deployment of such large models presents significant challenges. The computational resources required for inference, especially the key-value cache in transformer models, can be substantial. Researchers are actively exploring techniques to optimize these models, including quantization, pruning, and efficient attention mechanisms. KVCache reuse is one promising approach that can significantly reduce inference latency by avoiding redundant computations.

The field continues to evolve rapidly, with new architectures and training techniques being developed. From GPT to BERT, from T5 to modern multimodal models, the landscape of AI is constantly shifting. Understanding and optimizing these systems is crucial for making AI more accessible and practical for real-world applications."""

    # Repeat and concatenate to reach target length
    repeated_text = ""
    while len(tokenizer.encode(repeated_text)) < target_length:
        repeated_text += base_text + "\n\n"

    # Tokenize and trim to exact target length
    tokens = tokenizer.encode(repeated_text)[:target_length]
    return torch.tensor(tokens, dtype=torch.long)


def benchmark_full_prefill(model, tokenizer, input_tokens, device='cuda'):
    """
    Benchmark full prefill (rate=1)
    """
    print("\n" + "="*80)
    print("Method 1: Full Prefill (Baseline)")
    print("="*80)

    input_length = len(input_tokens)
    inputs = input_tokens.unsqueeze(0).to(device)

    # Create cache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=input_length + 100,
        device=device,
        dtype=model.dtype,
        passage_len=input_length
    )

    # Warm up
    with torch.no_grad():
        cache_position = torch.arange(input_length, device=device)
        inputs_embeds = model.model.embed_tokens(inputs).to(device)
        _ = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )
    torch.cuda.empty_cache()

    # Reset cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    # Actual benchmark - strict timing: from embed to end
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        cache_position = torch.arange(input_length, device=device)
        # Start timing right before embed
        inputs_embeds = model.model.embed_tokens(inputs).to(device)
        _ = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )

    torch.cuda.synchronize()
    prefill_time = time.time() - start_time

    print(f"\nResults:")
    print(f"  Input length: {input_length} tokens")
    print(f"  Prefill time: {prefill_time:.4f}s")
    print(f"  Throughput: {input_length/prefill_time:.2f} tokens/s")

    return {
        'method': 'Full Prefill',
        'input_length': input_length,
        'prefill_time': prefill_time,
        'throughput': input_length / prefill_time
    }


def benchmark_kvcache_reuse_random(model, tokenizer, input_tokens, recompute_rate=0.3, num_chunks=10, device='cuda'):
    """
    Benchmark KVCache reuse with random chunk selection

    Strategy:
    1. Split 16K tokens into num_chunks chunks
    2. Generate KV cache for all chunks
    3. Randomly select recompute_rate of chunks to recompute
    4. Load cached KV for remaining chunks
    5. Measure time for recomputing selected chunks + loading cached KV

    Args:
        recompute_rate: fraction of chunks to recompute (e.g., 0.3 = 30%)
        num_chunks: number of chunks to split input into
    """
    print("\n" + "="*80)
    print(f"Method 2: KVCache Reuse with Random Selection (Recompute {int(recompute_rate*100)}%)")
    print("="*80)

    input_length = len(input_tokens)
    chunk_size = input_length // num_chunks

    # Split into chunks
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else input_length
        chunks.append(input_tokens[start_idx:end_idx])

    print(f"\nSetup:")
    print(f"  Total input: {input_length} tokens")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Chunk size: ~{chunk_size} tokens")

    # Step 1: Generate KV cache for all chunks
    print(f"\nStep 1: Pre-generating KV cache for all chunks...")
    all_kv_caches = []

    for i, chunk in enumerate(chunks):
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=len(chunk) + 100,
            device=device,
            dtype=model.dtype,
            passage_len=len(chunk)
        )

        chunk_input = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            cache_position = torch.arange(len(chunk), device=device)
            inputs_embeds = model.model.embed_tokens(chunk_input).to(device)
            _ = model(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True
            )

        # Extract KV cache
        key_cache = torch.stack([past_key_values.key_cache[i][:, :, :len(chunk), :]
                                 for i in range(len(past_key_values.key_cache))])
        value_cache = torch.stack([past_key_values.value_cache[i][:, :, :len(chunk), :]
                                   for i in range(len(past_key_values.value_cache))])

        all_kv_caches.append({
            'key': key_cache.cpu(),  # Move to CPU to simulate cache storage
            'value': value_cache.cpu(),
            'length': len(chunk)
        })

    print(f"  ✓ Generated KV cache for {num_chunks} chunks")

    # Step 2: Randomly select chunks to recompute
    num_recompute = max(1, int(num_chunks * recompute_rate))
    recompute_indices = random.sample(range(num_chunks), num_recompute)
    recompute_indices.sort()
    cache_indices = [i for i in range(num_chunks) if i not in recompute_indices]

    print(f"\nStep 2: Random selection:")
    print(f"  Recompute chunks: {recompute_indices} ({num_recompute} chunks, {num_recompute/num_chunks*100:.1f}%)")
    print(f"  Reuse cache chunks: {cache_indices} ({len(cache_indices)} chunks, {len(cache_indices)/num_chunks*100:.1f}%)")

    recompute_tokens = sum([len(chunks[i]) for i in recompute_indices])
    cache_tokens = sum([len(chunks[i]) for i in cache_indices])
    print(f"  Recompute tokens: {recompute_tokens} ({recompute_tokens/input_length*100:.1f}%)")
    print(f"  Cache tokens: {cache_tokens} ({cache_tokens/input_length*100:.1f}%)")

    # Step 3: Warm up
    print(f"\nStep 3: Warming up...")
    combined_past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=input_length + 100,
        device=device,
        dtype=model.dtype,
        passage_len=input_length
    )

    # Load cached KV
    current_pos = 0
    for chunk_idx in cache_indices:
        cache = all_kv_caches[chunk_idx]
        chunk_len = cache['length']
        for layer_idx in range(len(combined_past_key_values.key_cache)):
            combined_past_key_values.key_cache[layer_idx][:, :, current_pos:current_pos+chunk_len, :] = \
                cache['key'][layer_idx].to(device)
            combined_past_key_values.value_cache[layer_idx][:, :, current_pos:current_pos+chunk_len, :] = \
                cache['value'][layer_idx].to(device)
            combined_past_key_values.past_tokens[layer_idx] += chunk_len
        current_pos += chunk_len

    # Recompute selected chunks
    for chunk_idx in recompute_indices:
        chunk = chunks[chunk_idx]
        chunk_input = chunk.unsqueeze(0).to(device)
        chunk_len = len(chunk)
        with torch.no_grad():
            cache_position = torch.arange(current_pos, current_pos + chunk_len, device=device)
            inputs_embeds = model.model.embed_tokens(chunk_input).to(device)
            _ = model(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                past_key_values=combined_past_key_values,
                return_dict=False,
                use_cache=True,
                dense=2
            )
        current_pos += chunk_len

    torch.cuda.empty_cache()

    # Step 4: Actual benchmark - strict timing
    print(f"\nStep 4: Running actual benchmark...")
    print(f"  (Timing includes: KV cache loading + recomputation)")

    # Create new cache for actual benchmark
    combined_past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=input_length + 100,
        device=device,
        dtype=model.dtype,
        passage_len=input_length
    )

    torch.cuda.synchronize()
    start_time = time.time()

    # Load cached KV to GPU (this is part of the "reuse" cost)
    current_pos = 0
    for chunk_idx in cache_indices:
        cache = all_kv_caches[chunk_idx]
        chunk_len = cache['length']
        for layer_idx in range(len(combined_past_key_values.key_cache)):
            combined_past_key_values.key_cache[layer_idx][:, :, current_pos:current_pos+chunk_len, :] = \
                cache['key'][layer_idx].to(device)
            combined_past_key_values.value_cache[layer_idx][:, :, current_pos:current_pos+chunk_len, :] = \
                cache['value'][layer_idx].to(device)
            combined_past_key_values.past_tokens[layer_idx] += chunk_len
        current_pos += chunk_len

    # Recompute selected chunks (strict timing: from embed to end)
    with torch.no_grad():
        for chunk_idx in recompute_indices:
            chunk = chunks[chunk_idx]
            chunk_input = chunk.unsqueeze(0).to(device)
            chunk_len = len(chunk)

            cache_position = torch.arange(current_pos, current_pos + chunk_len, device=device)
            # Timing includes embed
            inputs_embeds = model.model.embed_tokens(chunk_input).to(device)
            _ = model(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                past_key_values=combined_past_key_values,
                return_dict=False,
                use_cache=True,
                dense=2
            )
            current_pos += chunk_len

    torch.cuda.synchronize()
    prefill_time = time.time() - start_time

    print(f"\nResults:")
    print(f"  Total time (cache loading + recomputation): {prefill_time:.4f}s")
    print(f"  Effective throughput: {input_length/prefill_time:.2f} tokens/s")
    print(f"  (Note: Throughput accounts for processing all {input_length} tokens)")

    return {
        'method': f'KVCache Reuse Random ({int(recompute_rate*100)}%)',
        'input_length': input_length,
        'recompute_tokens': recompute_tokens,
        'cache_tokens': cache_tokens,
        'num_chunks': num_chunks,
        'recompute_chunks': num_recompute,
        'prefill_time': prefill_time,
        'effective_throughput': input_length / prefill_time
    }


def main():
    """
    Main benchmark function
    """
    # Configuration
    model_path = '/mnt/data/models/openPangu-Embedded-1B-V1.1/'
    target_length = 7000
    recompute_rate = 0.3
    num_chunks = 10
    device = 'cuda:0'
    random_seed = 42

    random.seed(random_seed)

    print("="*80)
    print("KVCache Reuse Benchmark for openPangu")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Target length: {target_length} tokens")
    print(f"  Recompute rate: {int(recompute_rate*100)}%")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Random seed: {random_seed}")
    print(f"  Device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    torch.set_default_dtype(config.torch_dtype)
    with torch.no_grad():
        model = PanguEmbeddedForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=config.torch_dtype
        )
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded")

    # Generate input text
    print(f"\nGenerating {target_length}-token input text...")
    input_tokens = generate_long_text(tokenizer, target_length)
    actual_length = len(input_tokens)
    print(f"  ✓ Generated {actual_length} tokens")

    # Run benchmarks
    results = []

    # Method 1: Full prefill
    result_full = benchmark_full_prefill(model, tokenizer, input_tokens, device)
    results.append(result_full)

    torch.cuda.empty_cache()

    # Method 2: KVCache reuse with random selection
    result_reuse = benchmark_kvcache_reuse_random(
        model, tokenizer, input_tokens, recompute_rate, num_chunks, device
    )
    results.append(result_reuse)

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    full_time = result_full['prefill_time']
    reuse_time = result_reuse['prefill_time']
    speedup = full_time / reuse_time
    time_saved = full_time - reuse_time
    percent_saved = (time_saved / full_time) * 100

    actual_recompute_rate = result_reuse['recompute_tokens'] / result_reuse['input_length']
    expected_time_saved = (1 - actual_recompute_rate) * 100

    print(f"\nFull Prefill (Baseline):")
    print(f"  Time: {full_time:.4f}s")
    print(f"  Throughput: {result_full['throughput']:.2f} tokens/s")

    print(f"\nKVCache Reuse (Random {int(recompute_rate*100)}% recompute):")
    print(f"  Time: {reuse_time:.4f}s")
    print(f"  Cache tokens: {result_reuse['cache_tokens']} ({result_reuse['cache_tokens']/actual_length*100:.1f}%)")
    print(f"  Recompute tokens: {result_reuse['recompute_tokens']} ({actual_recompute_rate*100:.1f}%)")
    print(f"  Effective Throughput: {result_reuse['effective_throughput']:.2f} tokens/s")

    print(f"\nSpeedup Analysis:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_saved:.4f}s ({percent_saved:.1f}%)")
    print(f"  Expected time saved (theory): ~{expected_time_saved:.1f}%")
    print(f"  Actual vs Expected: {percent_saved:.1f}% vs {expected_time_saved:.1f}%")
    print(f"  Efficiency: {percent_saved/expected_time_saved*100:.1f}%")

    if percent_saved >= expected_time_saved * 0.7:  # Within 70% of theoretical
        print(f"\n✓ SUCCESS: KVCache reuse achieves good speedup!")
    else:
        print(f"\n⚠ Note: Speedup lower than expected (overhead from cache operations)")

    print("="*80)

    # Save results
    output_file = f"/mnt/data/processCache/benchmark_results_pangu_{actual_length}tokens_random_recompute{int(actual_recompute_rate*100)}.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KVCache Reuse Benchmark Results (Random Selection)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Model: {model_path}\n")
        f.write(f"  Input length: {actual_length} tokens\n")
        f.write(f"  Recompute rate: {int(actual_recompute_rate*100)}%\n")
        f.write(f"  Number of chunks: {num_chunks}\n")
        f.write(f"  Random seed: {random_seed}\n\n")
        f.write(f"Full Prefill:\n")
        f.write(f"  Time: {full_time:.4f}s\n")
        f.write(f"  Throughput: {result_full['throughput']:.2f} tokens/s\n\n")
        f.write(f"KVCache Reuse:\n")
        f.write(f"  Time: {reuse_time:.4f}s\n")
        f.write(f"  Cache tokens: {result_reuse['cache_tokens']}\n")
        f.write(f"  Recompute tokens: {result_reuse['recompute_tokens']}\n")
        f.write(f"  Effective Throughput: {result_reuse['effective_throughput']:.2f} tokens/s\n\n")
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Time saved: {time_saved:.4f}s ({percent_saved:.1f}%)\n")
        f.write(f"Expected vs Actual: {expected_time_saved:.1f}% vs {percent_saved:.1f}%\n")
        f.write(f"Efficiency: {percent_saved/expected_time_saved*100:.1f}%\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
