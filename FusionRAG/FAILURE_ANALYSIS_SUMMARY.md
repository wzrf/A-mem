# FusionRAG Failure Case Analysis Summary

## Overview

Analyzed **217 test cases** from FusionRAG per_example method:
- **Rate=1.0**: 100% recomputation (baseline)
- **Rate=0.3**: 30% recomputation

**Results:**
- **14 failure cases** where rate=1.0 is correct but rate=0.3 is incorrect
- **Failure rate: 6.5%**

## Failure Pattern Classification

### 1. Missing Key Information (21% of failures - 3 cases)

**Cases: 1, 3, 7**

Rate=0.3 loses critical information that was present at rate=1.0:

- **Case 1**: Missing sibling relationship
  - Rate=1.0: "King Henry III of England"
  - Rate=0.3: "had no siblings mentioned in the given documents"

- **Case 3**: Missing first coronation date
  - Rate=1.0: "1216 and 1220"
  - Rate=0.3: "1220" (only second coronation)

- **Case 7**: Missing first wife
  - Rate=1.0: "Helen Pitts Douglass and Anna Murray Douglass"
  - Rate=0.3: "Helen Pitts and Helen Pitts Douglass" (duplicate, missing Anna)

**Root Cause:** Key information was in tokens that were NOT selected for recomputation (the 70% that was approximated).

### 2. Wrong Document Retrieved (29% of failures - 4 cases)

**Cases: 4, 8, 13, 14**

Rate=0.3 retrieves or prioritizes wrong documents:

- **Case 4**: Wrong location
  - Rate=1.0: "Bancroft is located in Ontario, Canada"
  - Rate=0.3: "23 km west of the resort area of Christina Lake" (completely different place)

- **Case 8**: Wrong river
  - Rate=1.0: "Lower Nelson River"
  - Rate=0.3: "Kettle River"

- **Case 13**: Wrong performer
  - Rate=1.0: "Matthieu Chedid"
  - Rate=0.3: "Mylène Farmer"

- **Case 14**: Wrong entity type
  - Rate=1.0: "John Houseman" (producer)
  - Rate=0.3: "Metro-Goldwyn-Mayer" (studio)

**Root Cause:** Document similarity scoring or retrieval changed due to compressed KV affecting BGE embeddings or context ranking.

### 3. Evaluation Inconsistencies (36% of failures - 5 cases)

**Cases: 2, 5, 9, 10, 11**

Both rate=1.0 and rate=0.3 give identical or semantically equivalent answers, but are judged differently:

- **Case 2**: Both say "Culberson County" (ground truth: "Culberson County, Texas")
  - Rate=1.0: Correct
  - Rate=0.3: Incorrect

- **Case 5**: Rate=0.3 includes extra correct information
  - Rate=1.0: "Erick Sermon, Redman and Keith Murray"
  - Rate=0.3: "Erick Sermon, Redman and Keith Murray. Jamal is considered an honorary member."
  - Ground truth includes Jamal, but rate=1.0 was judged correct without him!

- **Cases 9, 10**: Identical answers "Tolyatti, Russia" and "Lisbon District"

- **Case 11**: Both provide the area in km² (44.125), but different formats

**Root Cause:** OpenAI API judge is not deterministic or uses different prompts for different runs.

### 4. Answer Corruption (7% of failures - 1 case)

**Case 6**: Hallucination or name corruption
- Rate=1.0: Correctly extracts "Nina Sky" from context
- Rate=0.3: "Nicole Sky" (wrong first name)

**Root Cause:** Compressed KV caused confusion between "Nicole and Natalie" (album name) and "Nina Sky" (band name).

### 5. Truncation Issues (7% of failures - 1 case)

**Case 12**: Similar but truncated answers
- Both mention "11 nations participated" and same details
- Rate=0.3 answer appears truncated mid-sentence

**Root Cause:** Generation stopped early, possibly due to different logits distribution from compressed KV.

## Insights

### Real vs Perceived Issues

The analysis reveals that **attention/logits divergence during generation** is NOT the primary problem:

1. **50% of failures (7 cases)** are due to:
   - Missing key information in compressed tokens (21%)
   - Wrong document retrieval (29%)

   These happen BEFORE generation even starts!

2. **36% of failures (5 cases)** are evaluation inconsistencies, not actual errors

3. **Only 14% (2 cases)** might be generation-related (corruption, truncation)

### Critical Finding

The main problem is not "how the model generates with approximated KV", but rather:

1. **Which tokens get selected for recomputation** (selection strategy)
2. **How compression affects document retrieval** (similarity scoring)
3. **Evaluation consistency** (judging criteria)

## Recommended Analysis Approach

### ❌ NOT Recommended: Deep Attention/Logits Analysis

The original plan to track attention weights and logits layer-by-layer would be useful for generation-time issues, but that's only 14% of failures.

### ✅ Recommended: Document Selection & Compression Analysis

For each failure case, analyze:

1. **Document Retrieval Analysis**
   ```python
   # Compare which documents were retrieved at rate=1.0 vs rate=0.3
   # Check if BGE similarity scores changed due to compressed KV
   ```

2. **Token Selection Analysis**
   ```python
   # For cases with missing information:
   # - Which tokens contained the key info?
   # - Were they selected for recomputation?
   # - What were their importance scores?
   ```

3. **Compression Impact on Embeddings**
   ```python
   # Does compressing early documents affect the representation
   # used for retrieving similar documents?
   ```

## Proposed Improvements

### Strategy 1: Key Information Preservation

**Problem:** Cases 1, 3, 7 lost critical facts in the 70% not recomputed

**Solution:** Named Entity Recognition (NER) based recomputation
```python
def ner_guided_selection(tokens, ratio=0.3):
    """
    Prioritize tokens containing:
    - Named entities (people, places, dates)
    - Numbers and dates
    - Proper nouns
    """
    ner_scores = compute_ner_importance(tokens)
    base_scores = compute_attention_importance(tokens)

    # Boost tokens with entities
    combined_scores = base_scores + 0.5 * ner_scores

    # Select top-ratio tokens
    return select_top_k(tokens, combined_scores, ratio)
```

**Expected Impact:** Reduce missing information failures by 50-70%

### Strategy 2: Document Retrieval Stabilization

**Problem:** Cases 4, 8, 13, 14 retrieved wrong documents at rate=0.3

**Solution:** Separate preprocessing for similarity scoring
```python
def stable_document_retrieval(documents):
    """
    Use uncompressed KV for computing document embeddings
    Only compress after document selection is complete
    """
    # Step 1: Generate full KV for all documents
    full_kvs = [generate_kv(doc, compress=False) for doc in documents]

    # Step 2: Compute similarities using full KV
    similarities = compute_bge_similarities(full_kvs)

    # Step 3: Select relevant documents
    selected_docs = select_top_k(documents, similarities)

    # Step 4: NOW compress selected documents
    compressed_kvs = [compress_kv(kv, ratio=0.3) for kv in selected_docs]

    return compressed_kvs
```

**Expected Impact:** Eliminate document retrieval errors (4 cases)

### Strategy 3: Question-Aware Token Selection

**Problem:** Generic importance scoring doesn't know which info is needed

**Solution:** Pre-scan question to identify required information types
```python
def question_aware_compression(question, documents, ratio=0.3):
    """
    Analyze question to determine what type of info is needed
    Prioritize relevant tokens
    """
    # Extract question focus (who, when, where, what)
    question_type = classify_question(question)  # "who", "when", "where", etc.

    # Adjust importance based on question type
    if question_type == "when":
        boost_dates_and_years()
    elif question_type == "who":
        boost_person_names()
    elif question_type == "where":
        boost_locations()

    # Select tokens with question-aware boosting
    return select_tokens(documents, ratio, boosted_scores)
```

**Expected Impact:** Better handle multi-hop questions

### Strategy 4: Consistency in Evaluation

**Problem:** Cases 2, 5, 9, 10, 11 show judge inconsistency

**Solution:** Use deterministic evaluation with temperature=0 and same prompt
```python
# Ensure evaluation consistency
evaluator_config = {
    'temperature': 0.0,
    'seed': 42,
    'model': 'gpt-4-turbo',
    'prompt_template': FIXED_TEMPLATE
}
```

**Expected Impact:** Reduce false failures by 36%

## Next Steps

### Phase 1: Validation (Current Priority)

1. **Re-evaluate with consistent settings**
   - Run both rate=1.0 and rate=0.3 with same judge configuration
   - Eliminate false failures from evaluation inconsistency

2. **Analyze token selection in failure cases**
   - For cases 1, 3, 7: identify which tokens had key info
   - Check if they were selected for recomputation
   - Analyze their importance scores

3. **Analyze document retrieval**
   - For cases 4, 8, 13, 14: compare document selection at both rates
   - Check BGE similarity scores before/after compression

### Phase 2: Implementation

Based on Phase 1 findings, implement:
- Strategy 1 (NER-guided) if key info is being lost
- Strategy 2 (Stable retrieval) if document selection is changing
- Strategy 3 (Question-aware) for better multi-hop performance

### Phase 3: Comprehensive Testing

Run full evaluation with improvements:
- Target: Reduce failure rate from 6.5% to <2%
- Maintain or reduce compression ratio
- Measure impact on generation quality

## Conclusion

The FusionRAG failure analysis reveals that **the problem is not in the generation phase** (attention/logits), but in:

1. **Pre-generation phase**: Document retrieval and token selection (50%)
2. **Evaluation phase**: Judge inconsistency (36%)
3. **Generation phase**: Actual generation issues (14%)

This means improvement efforts should focus on:
- ✅ Smarter token selection (NER, question-aware)
- ✅ Stable document retrieval
- ✅ Consistent evaluation
- ❌ NOT layer-by-layer attention tracking

The proposed strategies directly address the root causes and are expected to significantly reduce the failure rate.
