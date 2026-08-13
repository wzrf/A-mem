# KV Cache Comparison Analysis: Rate=0.3 vs Rate=1.0

## Example 4, Sub-question 0

### Question Context
- **Main Question**: "When was the sibling of Alice de Lusignan, Countess of Surrey crowned?"
- **Sub-question**: "Who were the siblings of Alice de Lusignan, Countess of Surrey?"
- **Ground Truth**: "Alice de Lusignan had a uterine half-brother, King Henry III of England."
- **Generated Answer** (5 tokens): " King Henry III of England"

---

## Key Findings

### 1. **Rate=0.3 Actually Outperformed Rate=1.0 at First Token**

**Decode Step 1** (Position 1998):
- **Rate=1.0**: " John" (prob=0.794) ❌ WRONG
- **Rate=0.3**: " Henry" (prob=0.998) ✓ CORRECT
- **Logits L2**: 464.13
- **Probability L2**: 1.286

**Insight**: Despite only recomputing 30% of the context, rate=0.3 made the CORRECT prediction with much higher confidence (99.8% vs 79.4%). This suggests that selective recomputation successfully preserved the critical information needed to identify "Henry" rather than "John".

This is a counter-intuitive result showing that aggressive compression doesn't necessarily hurt - and can sometimes help - accuracy.

---

### 2. **Logits Divergence ≠ Prediction Divergence**

| Step | Logits L2 | Probs L2 | Top-1 Changed? | Insight |
|------|-----------|----------|----------------|---------|
| 1 | 464.13 | 1.286 | ✓ Yes | Large divergence in both logits and probs → different tokens |
| 2 | 475.01 | 0.000087 | ✗ No | Large logits L2 but tiny probs L2 → same token " III" |
| 3 | 745.41 | 1.251 | ✓ Yes | Largest logits L2, large probs L2 → different tokens |
| 4 | 600.62 | 0.000087 | ✗ No | Large logits L2 but tiny probs L2 → same token " England" |
| 5 | 1264.75 | 0.009 | ✗ No | Largest logits L2 but small probs L2 → same token "," |

**Insight**: Softmax normalization provides significant robustness. Even when raw logits differ substantially (L2 > 1200), the probability distributions can remain nearly identical if the relative ordering of top tokens is preserved.

Steps 2 and 4 show perfect agreement on high-confidence predictions (>99.9%) despite logits L2 distances of 475-600.

---

### 3. **Attention Patterns: Selective Head Impact**

The 28 query heads (4 KV head groups × 7 query heads) show **heterogeneous** sensitivity to compression:

**Decode Step 1** (when predictions diverged):
- Heads 0-6 (KV group 0): L2 ≈ 1.008
- Heads 7-13 (KV group 1): L2 ≈ 0.715 ← least affected
- Heads 14-20 (KV group 2): L2 ≈ 0.992
- Heads 21-27 (KV group 3): L2 ≈ 0.926
- **Overall L2**: 4.84

**Decode Step 2** (when predictions agreed):
- Heads 0-6: L2 ≈ 0.672
- Heads 7-13: L2 ≈ 0.715
- Heads 14-20: L2 ≈ 0.041 ← dramatically reduced!
- Heads 21-27: L2 ≈ 0.030 ← dramatically reduced!
- **Overall L2**: 2.59

**Decode Step 3** (when predictions diverged again):
- Heads 0-6: L2 ≈ 0.036 ← now very small
- Heads 7-13: L2 ≈ 0.453 ← now largest
- Heads 14-20: L2 ≈ 0.025
- Heads 21-27: L2 ≈ 0.033
- **Overall L2**: 1.21

**Insight**: Different head groups show different sensitivity patterns across decode steps:
- Some heads (14-27) adapt quickly and converge by step 2-3
- Other heads (7-13) maintain moderate differences
- The compression impact is **not uniform** across heads, suggesting different heads encode different types of information, and compression affects them differently

---

### 4. **Critical Decision Points**

**Step 1**: First substantive token
- Prediction divergence: " John" vs " Henry"
- HIGH attention divergence (overall L2 = 4.84)
- HIGH probability divergence (L2 = 1.286)
- **This is the most critical step** - sets the trajectory for the entire answer

**Step 3**: Syntactic continuation
- Prediction divergence: " of" vs ","
- LOWEST attention divergence (overall L2 = 1.21) despite token change
- Moderate probability divergence (L2 = 1.251)
- Shows that **low attention divergence doesn't guarantee identical predictions**

**Step 5**: Punctuation
- No prediction divergence (both predict ",")
- HIGHEST logits divergence (L2 = 1264.75)
- But very low probability divergence (L2 = 0.009)
- Shows **robustness to logits perturbations** for high-confidence predictions

---

## Interpretation

### What Does This Tell Us About FusionRAG's 30% Recomputation?

1. **Information Preservation**: The selective recomputation strategy (rate=0.3) successfully preserved the critical information needed to identify "Henry III" over alternatives like "John". This suggests the token selection algorithm effectively identifies and retains semantically important context.

2. **Prediction Stability**: In 3 out of 5 decode steps, both configurations predicted the same top-1 token despite different KV cache compression rates. The model shows significant robustness to attention pattern perturbations.

3. **Attention Dynamics**: The varying L2 distances across head groups suggest that:
   - Different attention heads specialize in different aspects of context
   - Compression affects heads non-uniformly
   - Some heads are more resilient to compression than others

4. **Softmax Normalization as Stabilizer**: Large logits differences (600-1200 L2) often result in minimal probability differences (<0.01 L2) when the top predictions remain aligned. This built-in stability is crucial for compression's viability.

5. **First Token is Critical**: The highest impact is at the first substantive token (step 1), where the answer trajectory is established. Interestingly, rate=0.3 performed better here, suggesting compression might reduce noise or overfitting to less relevant context.

---

## Open Questions

1. **Why did rate=0.3 outperform rate=1.0 at step 1?**
   - Did full recomputation introduce noise from irrelevant passages?
   - Did selective recomputation focus attention on the most relevant information?

2. **What determines which heads are most affected by compression?**
   - Are certain head groups more sensitive to local vs global context?
   - Can we identify which heads are "compression-critical"?

3. **Is there a pattern in when compression helps vs hurts?**
   - Step 1: compression helped (correct token)
   - Step 3: compression hurt (grammatical difference)
   - Is there a way to predict this?

---

## Recommendations for Further Analysis

1. **Analyze more examples** to see if the "step 1 advantage" for rate=0.3 is consistent
2. **Examine which tokens were selected for recomputation** in rate=0.3 to understand the selection strategy
3. **Correlate head-level divergence with semantic roles** (e.g., are certain heads responsible for entity recognition?)
4. **Investigate the "compression helps" phenomenon** - when does reducing context improve accuracy?
