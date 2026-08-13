#!/usr/bin/env python3
"""
Analyze FusionRAG failure cases:
Compare cases where rate=0.3 fails but rate=1.0 succeeds

This script will:
1. Load test results for rate=1.0 and rate=0.3
2. Identify failure cases (correct at rate=1, wrong at rate=0.3)
3. Re-run these cases with attention and logits tracking
4. Analyze the differences to identify root causes
"""

import json
import csv
import torch
import numpy as np
from typing import List, Dict, Tuple
import os
import sys
from collections import defaultdict

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)


def load_results(csv_file: str) -> Dict[str, Dict]:
    """Load test results from CSV file"""
    results = {}

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return results

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use (main_question, sub_question) as key
            # Try both lowercase and capitalized column names
            main_q = row.get('main_question', row.get('Main Question', ''))
            sub_q = row.get('sub_question', row.get('Sub Question', ''))
            key = (main_q, sub_q)
            results[key] = row

    return results


def find_failure_cases(
    rate1_results: Dict,
    rate03_results: Dict
) -> List[Tuple[str, str]]:
    """
    Find cases where:
    - rate=1.0 is correct
    - rate=0.3 is incorrect

    Returns list of (main_question, sub_question) tuples
    """
    failure_cases = []

    for key in rate1_results:
        if key not in rate03_results:
            continue

        # Try both lowercase and capitalized column names
        r1_correct = rate1_results[key].get('correct', rate1_results[key].get('Correct', '')).lower() == 'true'
        r03_correct = rate03_results[key].get('correct', rate03_results[key].get('Correct', '')).lower() == 'true'

        # Found a failure case
        if r1_correct and not r03_correct:
            failure_cases.append(key)

    return failure_cases


class AttentionLogitsTracker:
    """
    Track attention scores and logits during generation

    This will be used to compare:
    - Attention patterns between rate=1.0 and rate=0.3
    - Logits at each layer
    - Token probability distributions
    """

    def __init__(self):
        self.attention_scores = []  # List of attention matrices per layer
        self.logits = []  # List of logits per layer
        self.hidden_states = []  # Hidden states at each layer
        self.layer_outputs = {}  # Store outputs per layer

    def reset(self):
        """Reset all tracked data"""
        self.attention_scores = []
        self.logits = []
        self.hidden_states = []
        self.layer_outputs = {}

    def register_hooks(self, model):
        """
        Register forward hooks to track attention and logits

        This needs to be implemented based on the specific model architecture
        """
        hooks = []

        # Hook for attention modules
        def attention_hook(module, input, output):
            # Store attention weights
            # Format depends on model architecture
            if hasattr(output, 'attentions'):
                self.attention_scores.append(output.attentions.detach().cpu())

        # Hook for each layer
        def layer_hook(module, input, output, layer_idx):
            # Store layer outputs
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            self.layer_outputs[layer_idx] = {
                'hidden_state': hidden_state.detach().cpu(),
                'input': input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu()
            }

        # Register hooks for all layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for idx, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(
                    lambda m, i, o, idx=idx: layer_hook(m, i, o, idx)
                )
                hooks.append(hook)

        return hooks

    def remove_hooks(self, hooks):
        """Remove all registered hooks"""
        for hook in hooks:
            hook.remove()

    def analyze_attention_diff(
        self,
        attn_rate1: torch.Tensor,
        attn_rate03: torch.Tensor
    ) -> Dict:
        """
        Analyze differences in attention patterns

        Args:
            attn_rate1: Attention scores from rate=1.0
            attn_rate03: Attention scores from rate=0.3

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # Compute attention distribution differences
        if attn_rate1 is not None and attn_rate03 is not None:
            # Shape: [batch, num_heads, seq_len, seq_len]

            # KL divergence between attention distributions
            attn1_flat = attn_rate1.flatten()
            attn03_flat = attn_rate03.flatten()

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            attn1_flat = attn1_flat + eps
            attn03_flat = attn03_flat + eps

            # Normalize to probability distributions
            attn1_flat = attn1_flat / attn1_flat.sum()
            attn03_flat = attn03_flat / attn03_flat.sum()

            # KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
            kl_div = (attn1_flat * torch.log(attn1_flat / attn03_flat)).sum().item()

            analysis['kl_divergence'] = kl_div

            # L1 distance
            l1_dist = (attn1_flat - attn03_flat).abs().sum().item()
            analysis['l1_distance'] = l1_dist

            # L2 distance
            l2_dist = ((attn1_flat - attn03_flat) ** 2).sum().sqrt().item()
            analysis['l2_distance'] = l2_dist

            # Find positions with largest attention changes
            diff = (attn_rate1 - attn_rate03).abs()
            top_changes = torch.topk(diff.flatten(), k=10)
            analysis['top_changes'] = top_changes.values.tolist()

        return analysis

    def analyze_logits_diff(
        self,
        logits_rate1: torch.Tensor,
        logits_rate03: torch.Tensor,
        tokenizer
    ) -> Dict:
        """
        Analyze differences in logits and predictions

        Args:
            logits_rate1: Logits from rate=1.0 [seq_len, vocab_size]
            logits_rate03: Logits from rate=0.3 [seq_len, vocab_size]
            tokenizer: Tokenizer for decoding

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        if logits_rate1 is not None and logits_rate03 is not None:
            # Convert to probabilities
            probs1 = torch.softmax(logits_rate1, dim=-1)
            probs03 = torch.softmax(logits_rate03, dim=-1)

            # Get top predictions
            top1_pred1 = torch.argmax(probs1, dim=-1)
            top1_pred03 = torch.argmax(probs03, dim=-1)

            # Find where predictions differ
            diff_positions = (top1_pred1 != top1_pred03).nonzero(as_tuple=True)[0]

            analysis['num_diff_predictions'] = len(diff_positions)
            analysis['diff_positions'] = diff_positions.tolist()[:20]  # First 20

            # Analyze probability shifts at differing positions
            if len(diff_positions) > 0:
                pos_analysis = []

                for pos in diff_positions[:10]:  # Analyze first 10
                    pos = pos.item()

                    # Get top-5 predictions for each
                    top5_1 = torch.topk(probs1[pos], k=5)
                    top5_03 = torch.topk(probs03[pos], k=5)

                    # Decode tokens
                    tokens1 = [tokenizer.decode([idx.item()]) for idx in top5_1.indices]
                    tokens03 = [tokenizer.decode([idx.item()]) for idx in top5_03.indices]

                    pos_analysis.append({
                        'position': pos,
                        'rate1_top5': list(zip(tokens1, top5_1.values.tolist())),
                        'rate03_top5': list(zip(tokens03, top5_03.values.tolist())),
                        'rate1_pred': tokenizer.decode([top1_pred1[pos].item()]),
                        'rate03_pred': tokenizer.decode([top1_pred03[pos].item()]),
                    })

                analysis['position_details'] = pos_analysis

            # Overall probability distribution difference
            kl_div_per_pos = []
            for pos in range(min(probs1.shape[0], probs03.shape[0])):
                p1 = probs1[pos]
                p03 = probs03[pos]

                # KL divergence
                eps = 1e-10
                kl = (p1 * torch.log((p1 + eps) / (p03 + eps))).sum().item()
                kl_div_per_pos.append(kl)

            analysis['avg_kl_divergence'] = np.mean(kl_div_per_pos)
            analysis['max_kl_divergence'] = np.max(kl_div_per_pos)
            analysis['kl_divergence_per_position'] = kl_div_per_pos[:50]  # First 50

        return analysis


def main():
    """
    Main analysis pipeline
    """
    print("="*80)
    print("FusionRAG Failure Case Analysis")
    print("="*80)

    # Configuration
    csv_path = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"  # Adjust this to actual path
    reprocess_method = "FusionRAG"

    rate1_csv = f"{csv_path}/FusionRAG_per_example_topk_10_rate_1_reevaluated.csv"
    rate03_csv = f"{csv_path}/FusionRAG_per_example_topk_10_rate_0.3_reevaluated.csv"

    # Load results
    print(f"\n1. Loading results...")
    print(f"   Rate=1.0: {rate1_csv}")
    print(f"   Rate=0.3: {rate03_csv}")

    rate1_results = load_results(rate1_csv)
    rate03_results = load_results(rate03_csv)

    print(f"   Loaded {len(rate1_results)} results for rate=1.0")
    print(f"   Loaded {len(rate03_results)} results for rate=0.3")

    if not rate1_results or not rate03_results:
        print("\n❌ Error: Could not load results. Please run tests first:")
        print("   python test_fusionrag_reflect.py --rate 1.0")
        print("   python test_fusionrag_reflect.py --rate 0.3")
        return

    # Find failure cases
    print(f"\n2. Identifying failure cases...")
    failure_cases = find_failure_cases(rate1_results, rate03_results)

    print(f"   Found {len(failure_cases)} cases where:")
    print(f"   - Rate=1.0 is CORRECT")
    print(f"   - Rate=0.3 is INCORRECT")

    if not failure_cases:
        print("\n✓ No failure cases found (all cases consistent)")
        return

    # Display sample failure cases
    print(f"\n3. Sample failure cases:")
    for i, (main_q, sub_q) in enumerate(failure_cases[:5]):
        print(f"\n   Case {i+1}:")
        print(f"   Main Q: {main_q[:80]}...")
        print(f"   Sub Q:  {sub_q[:80]}...")

        r1 = rate1_results[(main_q, sub_q)]
        r03 = rate03_results[(main_q, sub_q)]

        print(f"   Rate=1.0 answer: {r1.get('predicted', r1.get('Predicted', ''))[:100]}...")
        print(f"   Rate=0.3 answer: {r03.get('predicted', r03.get('Predicted', ''))[:100]}...")
        print(f"   Ground truth:    {r1.get('ground_truth', r1.get('Ground Truth', ''))[:100]}...")

    # Save failure cases for detailed analysis
    output_file = "failure_cases_analysis.json"
    print(f"\n4. Saving failure cases to {output_file}...")

    failure_data = []
    for main_q, sub_q in failure_cases:
        r1 = rate1_results[(main_q, sub_q)]
        r03 = rate03_results[(main_q, sub_q)]

        failure_data.append({
            'main_question': main_q,
            'sub_question': sub_q,
            'ground_truth': r1.get('ground_truth', r1.get('Ground Truth', '')),
            'rate1_predicted': r1.get('predicted', r1.get('Predicted', '')),
            'rate1_correct': r1.get('correct', r1.get('Correct', '')),
            'rate03_predicted': r03.get('predicted', r03.get('Predicted', '')),
            'rate03_correct': r03.get('correct', r03.get('Correct', '')),
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(failure_data, f, indent=2, ensure_ascii=False)

    print(f"   Saved {len(failure_data)} failure cases")

    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80)
    print(f"Total cases analyzed:     {len(rate1_results)}")
    print(f"Failure cases found:      {len(failure_cases)}")
    print(f"Failure rate:             {len(failure_cases)/len(rate1_results)*100:.1f}%")
    print("\nNext steps:")
    print("1. Run detailed attention/logits analysis on failure cases")
    print("2. Identify common patterns in failures")
    print("3. Propose improvements based on findings")
    print("="*80)


if __name__ == "__main__":
    main()
