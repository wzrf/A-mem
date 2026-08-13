#!/usr/bin/env python
"""
Unit test for Smart Query Selection functions
"""
import torch
import numpy as np
import sys
sys.path.insert(0, './ktransformers/util')

from utils import find_connected_components, smart_query_selection

def test_find_connected_components():
    """Test connected component finding"""
    print("Testing find_connected_components...")

    # Test 1: Simple case
    positions = [1, 2, 3, 7, 8, 12]
    components = find_connected_components(positions, max_gap=2)
    print(f"  Input: {positions}")
    print(f"  Output: {components}")
    assert len(components) == 3, f"Expected 3 components, got {len(components)}"
    assert components[0] == [1, 2, 3], f"First component wrong: {components[0]}"
    assert components[1] == [7, 8], f"Second component wrong: {components[1]}"
    assert components[2] == [12], f"Third component wrong: {components[2]}"
    print("  ✓ Test 1 passed")

    # Test 2: All connected
    positions = [1, 2, 3, 4, 5]
    components = find_connected_components(positions, max_gap=2)
    print(f"  Input: {positions}")
    print(f"  Output: {components}")
    assert len(components) == 1, f"Expected 1 component, got {len(components)}"
    print("  ✓ Test 2 passed")

    # Test 3: Empty input
    positions = []
    components = find_connected_components(positions, max_gap=2)
    assert components == [], f"Expected empty list, got {components}"
    print("  ✓ Test 3 passed")

    print("find_connected_components: All tests passed!\n")

def test_smart_query_selection():
    """Test smart query selection algorithm"""
    print("Testing smart_query_selection...")

    # Create synthetic attention scores with clear peaks
    doc_len = 100
    attention_scores = np.random.randn(doc_len) * 0.1  # Low baseline

    # Add high attention at specific positions (simulating "1216" and "1220" tokens)
    key_positions_1 = [20, 21, 22, 23]  # "1216"
    key_positions_2 = [50, 51, 52, 53]  # "1220"

    for p in key_positions_1:
        attention_scores[p] = 2.0 + np.random.randn() * 0.1
    for p in key_positions_2:
        attention_scores[p] = 1.8 + np.random.randn() * 0.1

    # Convert to tensor
    attn_tensor = torch.tensor(attention_scores, dtype=torch.float32)

    # Test with 30% selection
    target_ratio = 0.3
    system_len = 100

    selected = smart_query_selection(
        attention_scores=attn_tensor,
        doc_len=doc_len,
        target_ratio=target_ratio,
        system_len=system_len,
        device='cpu'
    )

    print(f"  Doc length: {doc_len}")
    print(f"  Target ratio: {target_ratio} ({int(doc_len * target_ratio)} tokens)")
    print(f"  Selected: {len(selected)} tokens")
    print(f"  Selected positions (first 20): {selected[:20]}")

    # Check that key positions are selected (adjusted for system_len offset)
    key_positions_global_1 = [p + system_len for p in key_positions_1]
    key_positions_global_2 = [p + system_len for p in key_positions_2]

    selected_key_1 = sum(1 for p in key_positions_global_1 if p in selected)
    selected_key_2 = sum(1 for p in key_positions_global_2 if p in selected)

    print(f"  Key positions 1 ('1216') selected: {selected_key_1}/{len(key_positions_1)}")
    print(f"  Key positions 2 ('1220') selected: {selected_key_2}/{len(key_positions_2)}")

    # At least some key positions should be selected
    assert selected_key_1 >= 2, f"Expected at least 2 key positions from group 1, got {selected_key_1}"
    assert selected_key_2 >= 2, f"Expected at least 2 key positions from group 2, got {selected_key_2}"

    # Check selection count is close to target
    expected_count = int(doc_len * target_ratio)
    assert abs(len(selected) - expected_count) <= 2, f"Selection count {len(selected)} not close to target {expected_count}"

    print("  ✓ All key positions selected!")
    print("smart_query_selection: All tests passed!\n")

def test_integration_with_importance_cache():
    """Test that the selection logic works with importance_cache-like structure"""
    print("Testing integration with importance_cache structure...")

    # Simulate importance_cache: [num_heads, doc_len]
    num_heads = 28
    doc_len = 200
    system_len = 100

    # Create attention scores for each head
    importance_cache = torch.randn(num_heads, doc_len) * 0.1

    # Add high attention at key positions for some heads
    key_positions = [30, 31, 32, 33, 80, 81, 82, 83]
    for h in range(num_heads):
        for p in key_positions:
            importance_cache[h, p] = 1.5 + torch.randn(1).item() * 0.2

    # Average across heads (as done in utils.py)
    attn_avg = importance_cache.mean(dim=0)  # [doc_len]

    # Apply smart selection
    selected = smart_query_selection(
        attention_scores=attn_avg,
        doc_len=doc_len,
        target_ratio=0.3,
        system_len=system_len,
        device='cpu'
    )

    print(f"  Num heads: {num_heads}")
    print(f"  Doc length: {doc_len}")
    print(f"  Selected: {len(selected)} tokens")

    # Check key positions are selected
    key_positions_global = [p + system_len for p in key_positions]
    selected_key = sum(1 for p in key_positions_global if p in selected)

    print(f"  Key positions selected: {selected_key}/{len(key_positions)}")
    assert selected_key >= 6, f"Expected at least 6 key positions, got {selected_key}"

    print("  ✓ Integration test passed!")
    print("Integration test: All tests passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Query Selection Unit Tests")
    print("=" * 60 + "\n")

    test_find_connected_components()
    test_smart_query_selection()
    test_integration_with_importance_cache()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
