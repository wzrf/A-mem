#!/usr/bin/env python3
"""
Deep Analysis Tool for FusionRAG

This tool provides detailed analysis of:
1. Attention score changes between rate=1.0 and rate=0.3
2. Logits differences at each layer
3. Hidden state divergence
4. Token probability distribution shifts

Usage:
    python deep_analysis_tool.py --case-file failure_cases_analysis.json --num-cases 5
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os


@dataclass
class LayerAnalysis:
    """Analysis results for a single layer"""
    layer_idx: int
    attention_kl_div: float
    logits_kl_div: float
    hidden_state_cosine_sim: float
    top_token_changes: List[Tuple[int, str, str]]  # position, token_r1, token_r03


class DeepAnalyzer:
    """
    Deep analyzer for comparing rate=1.0 and rate=0.3 generations

    This class hooks into the model to track:
    - Attention weights at each layer
    - Logits before and after each layer
    - Hidden states
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Storage for tracked data
        self.rate1_data = {}
        self.rate03_data = {}

        # Current tracking mode
        self.tracking_mode = None  # 'rate1' or 'rate03'

    def register_hooks(self):
        """
        Register forward hooks to capture intermediate outputs

        Returns hooks that need to be removed after analysis
        """
        hooks = []

        # Hook for each transformer layer
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Attention hook
            if hasattr(layer, 'self_attn'):
                def attn_hook(module, input, output, idx=layer_idx):
                    if self.tracking_mode:
                        # Extract attention weights if available
                        # This depends on whether output_attentions=True
                        if isinstance(output, tuple) and len(output) > 1:
                            attn_weights = output[1]  # Usually at index 1
                            self.store_attention(idx, attn_weights)

                hook = layer.self_attn.register_forward_hook(attn_hook)
                hooks.append(hook)

            # Layer output hook
            def layer_hook(module, input, output, idx=layer_idx):
                if self.tracking_mode:
                    if isinstance(output, tuple):
                        hidden_state = output[0]
                    else:
                        hidden_state = output

                    self.store_hidden_state(idx, hidden_state)

            hook = layer.register_forward_hook(layer_hook)
            hooks.append(hook)

        # Hook for final logits (lm_head)
        if hasattr(self.model, 'lm_head'):
            def logits_hook(module, input, output):
                if self.tracking_mode:
                    self.store_logits(output)

            hook = self.model.lm_head.register_forward_hook(logits_hook)
            hooks.append(hook)

        return hooks

    def store_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        """Store attention weights for later analysis"""
        key = f'layer_{layer_idx}_attention'

        if self.tracking_mode == 'rate1':
            self.rate1_data[key] = attn_weights.detach().cpu()
        elif self.tracking_mode == 'rate03':
            self.rate03_data[key] = attn_weights.detach().cpu()

    def store_hidden_state(self, layer_idx: int, hidden_state: torch.Tensor):
        """Store hidden state for later analysis"""
        key = f'layer_{layer_idx}_hidden'

        if self.tracking_mode == 'rate1':
            self.rate1_data[key] = hidden_state.detach().cpu()
        elif self.tracking_mode == 'rate03':
            self.rate03_data[key] = hidden_state.detach().cpu()

    def store_logits(self, logits: torch.Tensor):
        """Store final logits"""
        if self.tracking_mode == 'rate1':
            self.rate1_data['final_logits'] = logits.detach().cpu()
        elif self.tracking_mode == 'rate03':
            self.rate03_data['final_logits'] = logits.detach().cpu()

    def analyze_layer(self, layer_idx: int) -> Optional[LayerAnalysis]:
        """
        Analyze differences for a specific layer

        Returns LayerAnalysis object with detailed metrics
        """
        # Get attention weights
        attn_key = f'layer_{layer_idx}_attention'
        hidden_key = f'layer_{layer_idx}_hidden'

        attn1 = self.rate1_data.get(attn_key)
        attn03 = self.rate03_data.get(attn_key)

        hidden1 = self.rate1_data.get(hidden_key)
        hidden03 = self.rate03_data.get(hidden_key)

        if attn1 is None or attn03 is None:
            return None

        # Compute attention KL divergence
        attn_kl = self.compute_kl_divergence(
            self.flatten_attention(attn1),
            self.flatten_attention(attn03)
        )

        # Compute hidden state cosine similarity
        hidden_sim = 0.0
        if hidden1 is not None and hidden03 is not None:
            hidden_sim = F.cosine_similarity(
                hidden1.flatten(),
                hidden03.flatten(),
                dim=0
            ).item()

        # For logits, we need to compute per-position KL divergence
        logits_kl = 0.0
        # This will be computed separately for final logits

        return LayerAnalysis(
            layer_idx=layer_idx,
            attention_kl_div=attn_kl,
            logits_kl_div=logits_kl,
            hidden_state_cosine_sim=hidden_sim,
            top_token_changes=[]
        )

    def analyze_final_predictions(self) -> Dict:
        """
        Analyze final prediction differences

        Returns detailed breakdown of where and how predictions diverge
        """
        logits1 = self.rate1_data.get('final_logits')
        logits03 = self.rate03_data.get('final_logits')

        if logits1 is None or logits03 is None:
            return {}

        # Shape: [batch, seq_len, vocab_size]
        # Take first batch
        logits1 = logits1[0]
        logits03 = logits03[0]

        # Convert to probabilities
        probs1 = F.softmax(logits1, dim=-1)
        probs03 = F.softmax(logits03, dim=-1)

        # Get predictions
        pred1 = torch.argmax(probs1, dim=-1)
        pred03 = torch.argmax(probs03, dim=-1)

        # Find divergence points
        diff_mask = (pred1 != pred03)
        diff_positions = torch.nonzero(diff_mask, as_tuple=True)[0]

        analysis = {
            'num_positions': logits1.shape[0],
            'num_divergent': len(diff_positions),
            'divergence_rate': len(diff_positions) / logits1.shape[0] if logits1.shape[0] > 0 else 0,
            'divergent_positions': diff_positions.tolist()[:20],
            'position_details': []
        }

        # Analyze each divergent position
        for pos in diff_positions[:10]:  # First 10
            pos = pos.item()

            # Get top-k predictions
            k = 5
            top_k_1 = torch.topk(probs1[pos], k=k)
            top_k_03 = torch.topk(probs03[pos], k=k)

            # Decode tokens
            tokens1 = [self.tokenizer.decode([idx]) for idx in top_k_1.indices]
            tokens03 = [self.tokenizer.decode([idx]) for idx in top_k_03.indices]

            # Compute probability difference for predicted tokens
            pred_token_1 = pred1[pos].item()
            pred_token_03 = pred03[pos].item()

            prob_diff = (
                probs1[pos, pred_token_1].item() -
                probs03[pos, pred_token_03].item()
            )

            analysis['position_details'].append({
                'position': pos,
                'predicted_token_r1': self.tokenizer.decode([pred_token_1]),
                'predicted_token_r03': self.tokenizer.decode([pred_token_03]),
                'prob_r1': probs1[pos, pred_token_1].item(),
                'prob_r03': probs03[pos, pred_token_03].item(),
                'prob_diff': prob_diff,
                'top_k_r1': list(zip(tokens1, top_k_1.values.tolist())),
                'top_k_r03': list(zip(tokens03, top_k_03.values.tolist())),
                'kl_divergence': self.compute_kl_divergence(
                    probs1[pos],
                    probs03[pos]
                )
            })

        # Compute average KL divergence across all positions
        kl_divs = []
        for pos in range(min(probs1.shape[0], probs03.shape[0])):
            kl = self.compute_kl_divergence(probs1[pos], probs03[pos])
            kl_divs.append(kl)

        analysis['avg_kl_divergence'] = np.mean(kl_divs)
        analysis['max_kl_divergence'] = np.max(kl_divs)
        analysis['kl_divergence_distribution'] = {
            'mean': np.mean(kl_divs),
            'std': np.std(kl_divs),
            'min': np.min(kl_divs),
            'max': np.max(kl_divs),
            'median': np.median(kl_divs)
        }

        return analysis

    def compute_kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """
        Compute KL divergence: KL(P || Q) = sum(P * log(P/Q))

        Args:
            p: Probability distribution P
            q: Probability distribution Q

        Returns:
            KL divergence value
        """
        eps = 1e-10
        p = p + eps
        q = q + eps

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        kl = (p * torch.log(p / q)).sum().item()
        return kl

    def flatten_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Flatten attention tensor to probability distribution"""
        # Shape: [batch, num_heads, seq_len, seq_len]
        # Flatten and normalize
        flat = attn.flatten()
        flat = F.softmax(flat, dim=0)  # Normalize to probability
        return flat

    def generate_report(self, output_file: str = "deep_analysis_report.json"):
        """
        Generate comprehensive analysis report

        This includes:
        - Layer-wise analysis
        - Prediction divergence analysis
        - Recommendations for improvement
        """
        report = {
            'summary': {},
            'layer_analysis': [],
            'prediction_analysis': {},
            'recommendations': []
        }

        # Analyze each layer
        num_layers = len([k for k in self.rate1_data.keys() if 'layer_' in k and '_hidden' in k])

        print(f"\nAnalyzing {num_layers} layers...")

        for layer_idx in range(num_layers):
            layer_analysis = self.analyze_layer(layer_idx)
            if layer_analysis:
                report['layer_analysis'].append({
                    'layer': layer_idx,
                    'attention_kl_div': layer_analysis.attention_kl_div,
                    'hidden_state_cosine_sim': layer_analysis.hidden_state_cosine_sim
                })

        # Analyze final predictions
        print("Analyzing final predictions...")
        report['prediction_analysis'] = self.analyze_final_predictions()

        # Generate summary
        if report['layer_analysis']:
            attn_kls = [la['attention_kl_div'] for la in report['layer_analysis']]
            hidden_sims = [la['hidden_state_cosine_sim'] for la in report['layer_analysis']]

            report['summary'] = {
                'num_layers': num_layers,
                'avg_attention_kl': np.mean(attn_kls),
                'max_attention_kl': np.max(attn_kls),
                'layer_with_max_kl': int(np.argmax(attn_kls)),
                'avg_hidden_sim': np.mean(hidden_sims),
                'min_hidden_sim': np.min(hidden_sims),
                'layer_with_min_sim': int(np.argmin(hidden_sims))
            }

        # Generate recommendations based on analysis
        report['recommendations'] = self.generate_recommendations(report)

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to {output_file}")

        # Print summary
        self.print_summary(report)

        return report

    def generate_recommendations(self, report: Dict) -> List[str]:
        """
        Generate improvement recommendations based on analysis

        Args:
            report: Analysis report dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check which layers have highest divergence
        if report['layer_analysis']:
            layer_kls = [(la['layer'], la['attention_kl_div']) for la in report['layer_analysis']]
            layer_kls.sort(key=lambda x: x[1], reverse=True)

            top_divergent_layers = layer_kls[:3]

            if top_divergent_layers[0][1] > 1.0:  # High KL divergence
                recommendations.append(
                    f"⚠️  Layer {top_divergent_layers[0][0]} shows very high attention divergence "
                    f"(KL={top_divergent_layers[0][1]:.3f}). Consider:"
                    f"\n   - Increasing the recomputation ratio for this layer"
                    f"\n   - Using selective recomputation for attention-critical tokens"
                )

        # Check prediction divergence
        pred_analysis = report.get('prediction_analysis', {})
        div_rate = pred_analysis.get('divergence_rate', 0)

        if div_rate > 0.3:  # More than 30% positions diverge
            recommendations.append(
                f"⚠️  High prediction divergence rate ({div_rate:.1%}). Consider:"
                f"\n   - Increasing overall recomputation ratio"
                f"\n   - Implementing adaptive recomputation based on attention scores"
            )

        # Check KL divergence distribution
        kl_dist = pred_analysis.get('kl_divergence_distribution', {})
        max_kl = kl_dist.get('max', 0)

        if max_kl > 5.0:  # Very high KL at some positions
            recommendations.append(
                f"⚠️  Some positions have very high KL divergence (max={max_kl:.3f}). Consider:"
                f"\n   - Analyzing these specific positions for common patterns"
                f"\n   - Prioritizing recomputation at high-divergence positions"
            )

        # Check hidden state similarity
        summary = report.get('summary', {})
        min_sim = summary.get('min_hidden_sim', 1.0)

        if min_sim < 0.9:  # Low similarity
            layer_with_min = summary.get('layer_with_min_sim', 0)
            recommendations.append(
                f"⚠️  Layer {layer_with_min} has low hidden state similarity ({min_sim:.3f}). Consider:"
                f"\n   - Full recomputation for this layer"
                f"\n   - Investigating why this layer is sensitive to approximation"
            )

        if not recommendations:
            recommendations.append(
                "✓ Overall divergence is within acceptable range. "
                "Current recomputation strategy appears reasonable."
            )

        return recommendations

    def print_summary(self, report: Dict):
        """Print a human-readable summary of the analysis"""
        print("\n" + "="*80)
        print("DEEP ANALYSIS SUMMARY")
        print("="*80)

        summary = report.get('summary', {})
        if summary:
            print(f"\nLayer-wise Analysis ({summary['num_layers']} layers):")
            print(f"  Average attention KL divergence: {summary['avg_attention_kl']:.4f}")
            print(f"  Max attention KL divergence:     {summary['max_attention_kl']:.4f} (layer {summary['layer_with_max_kl']})")
            print(f"  Average hidden state similarity: {summary['avg_hidden_sim']:.4f}")
            print(f"  Min hidden state similarity:     {summary['min_hidden_sim']:.4f} (layer {summary['layer_with_min_sim']})")

        pred_analysis = report.get('prediction_analysis', {})
        if pred_analysis:
            print(f"\nPrediction Analysis:")
            print(f"  Total positions:      {pred_analysis['num_positions']}")
            print(f"  Divergent positions:  {pred_analysis['num_divergent']} ({pred_analysis['divergence_rate']:.1%})")
            print(f"  Average KL divergence: {pred_analysis['avg_kl_divergence']:.4f}")
            print(f"  Max KL divergence:     {pred_analysis['max_kl_divergence']:.4f}")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report.get('recommendations', []), 1):
            print(f"\n{i}. {rec}")

        print("\n" + "="*80)


def visualize_analysis(report_file: str):
    """
    Create visualizations from analysis report

    Args:
        report_file: Path to JSON report file
    """
    with open(report_file, 'r') as f:
        report = json.load(f)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FusionRAG Deep Analysis Visualization', fontsize=16)

    # Plot 1: Attention KL divergence per layer
    layer_analysis = report.get('layer_analysis', [])
    if layer_analysis:
        layers = [la['layer'] for la in layer_analysis]
        attn_kls = [la['attention_kl_div'] for la in layer_analysis]

        axes[0, 0].bar(layers, attn_kls)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Attention KL Divergence')
        axes[0, 0].set_title('Attention Divergence by Layer')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Hidden state similarity per layer
    if layer_analysis:
        hidden_sims = [la['hidden_state_cosine_sim'] for la in layer_analysis]

        axes[0, 1].plot(layers, hidden_sims, marker='o')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].set_title('Hidden State Similarity by Layer')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

    # Plot 3: Prediction divergence positions
    pred_analysis = report.get('prediction_analysis', {})
    if pred_analysis and pred_analysis.get('position_details'):
        positions = [pd['position'] for pd in pred_analysis['position_details']]
        kl_divs = [pd['kl_divergence'] for pd in pred_analysis['position_details']]

        axes[1, 0].scatter(positions, kl_divs, alpha=0.6)
        axes[1, 0].set_xlabel('Token Position')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].set_title('Prediction KL Divergence at Divergent Positions')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    summary_text = f"""
    Summary Statistics:

    Layers analyzed: {report['summary']['num_layers']}

    Avg attention KL: {report['summary']['avg_attention_kl']:.4f}
    Max attention KL: {report['summary']['max_attention_kl']:.4f}

    Avg hidden similarity: {report['summary']['avg_hidden_sim']:.4f}
    Min hidden similarity: {report['summary']['min_hidden_sim']:.4f}

    Prediction divergence rate: {pred_analysis.get('divergence_rate', 0):.1%}
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('deep_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to deep_analysis_visualization.png")


if __name__ == "__main__":
    print("Deep Analysis Tool for FusionRAG")
    print("="*80)
    print("\nThis tool needs to be integrated with FusionRAG test script.")
    print("Please see analyze_failure_cases.py for usage.")
