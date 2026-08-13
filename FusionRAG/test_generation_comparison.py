#!/usr/bin/env python3
"""
完整对比测试：比较 per_head_generation.py 和 test_fusionrag_reflect.py 的完整生成流程
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from transformers import AutoTokenizer, AutoConfig
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import smart_query_selection


def main():
    print("=" * 100)
    print("Generation Comparison Test")
    print("=" * 100)

    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    data_path = './result_reflect.json'
    cache_path = '/mnt/data/reflect/'
    model_name = 'Qwen2.5-7B-Instruct'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    device = 'cuda:0'

    example_idx = 4
    sub_question_idx = 1
    rate = 0.3
    max_new_tokens = 50

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load data
    print("\nLoading data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=example_idx + 1,
        preprocess=False
    )

    q_data = questions_data[example_idx]
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\nQuestion: {sub_q_info['query']}")
    print(f"Ground Truth: {sub_q_info['answer']}")

    # Build query tensor
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    query_tensor = torch.tensor(question_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Prepare cache
    doc_chunk_ids = [0] + sub_q_info['chunk_ids']
    full_cache_path = os.path.join(cache_path, model_name, 'kv_cache')

    # ========================================================================
    # Method 1: per_head_generation.py style (no sparse prefill, just query forward)
    # ========================================================================
    print("\n" + "=" * 100)
    print("Method 1: per_head_generation.py style (query forward only)")
    print("=" * 100)

    # Initialize cache
    past_key_values_1 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=32768,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Load KV cache
    passages_len = []
    prefix_len = 0

    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True)

        cache_len = chunk_key[0].shape[2]
        passages_len.append(cache_len)

        for layer_idx in range(model.config.num_hidden_layers):
            current_pos = past_key_values_1.past_tokens[layer_idx]
            past_key_values_1.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_key[layer_idx].to(device))
            past_key_values_1.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_value[layer_idx].to(device))
            past_key_values_1.past_tokens[layer_idx] += cache_len

        if chunk_id == 0:
            prefix_len = cache_len

    doc_len = sum(passages_len[1:])
    total_cache_len = prefix_len + doc_len

    print(f"Prefix: {prefix_len}, Doc: {doc_len}, Total: {total_cache_len}")

    # Generate (per_head style): forward query at correct positions
    query_len = query_tensor.shape[1]
    cache_position = torch.arange(prefix_len + doc_len, prefix_len + doc_len + query_len, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=query_tensor,
            past_key_values=past_key_values_1,
            cache_position=cache_position,
            return_dict=True,
            use_cache=True
        )
        logits = outputs.logits[0, -1, :]

    generated_ids_1 = []
    current_position = prefix_len + doc_len + query_len

    for step in range(max_new_tokens):
        next_token_id = torch.argmax(logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_ids_1.append(next_token_id.item())

        # Forward next token
        next_token_embeds = model.model.embed_tokens(next_token_id.unsqueeze(0).unsqueeze(0)).to(device)
        cache_position = torch.tensor([current_position], device=device)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_token_embeds,
                past_key_values=past_key_values_1,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, -1, :]

        current_position += 1

    generated_text_1 = tokenizer.decode(generated_ids_1, skip_special_tokens=True)
    print(f"\nGenerated (Method 1 - no sparse prefill): {generated_text_1}")

    # ========================================================================
    # Method 2: utils.py style (sparse reprocess)
    # ========================================================================
    print("\n" + "=" * 100)
    print("Method 2: utils.py style (sparse reprocess)")
    print("=" * 100)

    # Re-initialize cache
    past_key_values_2 = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=32768,
        device=device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Reload KV cache
    for chunk_id in doc_chunk_ids:
        chunk_key = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_key.pt', weights_only=True)
        chunk_value = torch.load(f'{full_cache_path}/{example_idx}_{chunk_id}_value.pt', weights_only=True)

        cache_len = chunk_key[0].shape[2]

        for layer_idx in range(model.config.num_hidden_layers):
            current_pos = past_key_values_2.past_tokens[layer_idx]
            past_key_values_2.key_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_key[layer_idx].to(device))
            past_key_values_2.value_cache[layer_idx].narrow(2, current_pos, cache_len).copy_(chunk_value[layer_idx].to(device))
            past_key_values_2.past_tokens[layer_idx] += cache_len

    # Compute attention and get selected indices
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    start_layer = num_layers * 3 // 4

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tensor)
        hidden_states = inputs_embeds

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = past_key_values_2.key_cache[layer_idx][:, :, :total_cache_len, :]

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)

            doc_attn = attn_weights[0, :, :, prefix_len:prefix_len + doc_len]
            doc_attn_avg = doc_attn.mean(dim=(0, 1)).cpu().float().numpy()

            layer_attention_scores[layer_idx] = doc_attn_avg

            # Continue forward
            value_states = past_key_values_2.value_cache[layer_idx][:, :, :total_cache_len, :]
            value_states = value_states.repeat_interleave(n_rep, dim=1)

            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    # Aggregate and select
    layers_to_use = list(range(start_layer, num_layers))
    multi_layer_attn = np.stack([layer_attention_scores[i] for i in layers_to_use]).mean(axis=0)

    selected_indices = smart_query_selection(
        attention_scores=multi_layer_attn,
        doc_len=doc_len,
        target_ratio=rate,
        system_len=prefix_len,
        device=device
    )

    print(f"Selected {len(selected_indices)} tokens ({len(selected_indices)/doc_len*100:.1f}%)")

    # Build reprocess indices
    k_need_index = sorted(selected_indices)
    query_positions = list(range(prefix_len + doc_len, prefix_len + doc_len + query_len))
    k_need_index.extend(query_positions)

    # Build all tokens
    passages = [system_tensor.to(device)]
    for chunk_id in doc_chunk_ids[1:]:
        doc_tensor = q_data['doc_tensors'][chunk_id - 1].to(device)
        passages.append(doc_tensor)
    passages.append(torch.tensor(question_tokens, device=device))

    all_tokens = torch.cat(passages)
    reprocess_tokens = all_tokens[k_need_index]
    reprocess_inputs = reprocess_tokens.unsqueeze(0)
    cache_position = torch.tensor(k_need_index, device=device)

    print(f"Reprocess: {len(k_need_index)} tokens")

    # Sparse reprocess
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(device)
        model_output = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values_2,
            return_dict=False,
            use_cache=True,
        )[0]

        logits = model_output[:, -1, :]

    # Generate
    generated_ids_2 = []
    current_position = prefix_len + doc_len + query_len

    for step in range(max_new_tokens):
        next_token_id = torch.argmax(logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_ids_2.append(next_token_id.item())

        next_token_embeds = model.model.embed_tokens(next_token_id.unsqueeze(0).unsqueeze(0)).to(device)
        cache_position = torch.tensor([current_position], device=device)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_token_embeds,
                past_key_values=past_key_values_2,
                cache_position=cache_position,
                return_dict=True,
                use_cache=True
            )
            logits = outputs.logits[0, -1, :]

        current_position += 1

    generated_text_2 = tokenizer.decode(generated_ids_2, skip_special_tokens=True)
    print(f"\nGenerated (Method 2 - sparse reprocess): {generated_text_2}")

    # ========================================================================
    # Compare
    # ========================================================================
    print("\n" + "=" * 100)
    print("Comparison")
    print("=" * 100)
    print(f"\nGround Truth: {sub_q_info['answer']}")
    print(f"\nMethod 1 (no sparse, query forward only): {generated_text_1}")
    print(f"Method 2 (sparse reprocess): {generated_text_2}")
    print(f"\nSame result: {generated_text_1 == generated_text_2}")


if __name__ == '__main__':
    main()
