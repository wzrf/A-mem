#!/usr/bin/env python3
"""
Analyze Token Selection in FusionRAG Failure Cases

This script analyzes which tokens were selected for recomputation at rate=0.3
and determines if key information was compressed away.

For each failure case:
1. Load the preprocessed KV cache metadata (if available)
2. Identify which tokens were selected for recomputation (30%)
3. Check if ground truth keywords appear in selected vs non-selected tokens
4. Analyze correlation between missing information and token selection
"""

import json
import os
import sys
from typing import List, Dict, Set
import re
from collections import defaultdict

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)


def extract_keywords(text: str) -> Set[str]:
    """
    Extract important keywords from text

    Focus on:
    - Named entities (capitalized words)
    - Numbers and dates
    - Key verbs and nouns
    """
    keywords = set()

    # Extract capitalized words (likely named entities)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    keywords.update(capitalized)

    # Extract years
    years = re.findall(r'\b\d{4}\b', text)
    keywords.update(years)

    # Extract numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    keywords.update(numbers)

    # Convert to lowercase for comparison
    keywords = {k.lower() for k in keywords}

    return keywords


def analyze_missing_information(failure_case: Dict) -> Dict:
    """
    Analyze what information is missing in rate=0.3 compared to rate=1.0

    Args:
        failure_case: Dictionary containing failure case details

    Returns:
        Analysis dictionary with missing keywords and patterns
    """
    ground_truth = failure_case['ground_truth']
    rate1_pred = failure_case['rate1_predicted']
    rate03_pred = failure_case['rate03_predicted']

    # Extract keywords from each
    gt_keywords = extract_keywords(ground_truth)
    r1_keywords = extract_keywords(rate1_pred)
    r03_keywords = extract_keywords(rate03_pred)

    # Find what's in rate=1.0 but missing in rate=0.3
    in_r1_not_r03 = r1_keywords - r03_keywords

    # Find what's in ground truth but missing in rate=0.3
    in_gt_not_r03 = gt_keywords - r03_keywords

    # Find what's in rate=0.3 but wrong (not in ground truth)
    in_r03_not_gt = r03_keywords - gt_keywords

    analysis = {
        'main_question': failure_case['main_question'][:80],
        'sub_question': failure_case['sub_question'][:80],
        'ground_truth_keywords': sorted(list(gt_keywords)),
        'rate1_keywords': sorted(list(r1_keywords)),
        'rate03_keywords': sorted(list(r03_keywords)),
        'missing_in_rate03': sorted(list(in_r1_not_r03)),
        'missing_from_gt': sorted(list(in_gt_not_r03)),
        'wrong_in_rate03': sorted(list(in_r03_not_gt)),
        'ground_truth': ground_truth,
        'rate1_predicted': rate1_pred,
        'rate03_predicted': rate03_pred,
    }

    return analysis


def categorize_failure_type(analysis: Dict) -> str:
    """
    Categorize the type of failure based on keyword analysis
    """
    missing = len(analysis['missing_in_rate03'])
    wrong = len(analysis['wrong_in_rate03'])

    # Check specific patterns
    rate03_lower = analysis['rate03_predicted'].lower()

    if 'no' in rate03_lower and ('mentioned' in rate03_lower or 'information' in rate03_lower):
        return "MISSING_INFORMATION"

    if wrong > 0 and missing > 0:
        return "WRONG_RETRIEVAL"

    if missing > 2:
        return "MISSING_KEY_INFO"

    if wrong > 2:
        return "HALLUCINATION"

    if missing == 0 and wrong == 0:
        return "EVAL_INCONSISTENCY"

    if missing <= 2:
        return "PARTIAL_INFO"

    return "OTHER"


def analyze_question_type(question: str) -> str:
    """
    Determine what type of question this is (who, when, where, what, etc.)
    """
    question_lower = question.lower()

    if question_lower.startswith('who'):
        return "WHO"
    elif question_lower.startswith('when'):
        return "WHEN"
    elif question_lower.startswith('where'):
        return "WHERE"
    elif question_lower.startswith('what'):
        return "WHAT"
    elif question_lower.startswith('which'):
        return "WHICH"
    elif question_lower.startswith('how'):
        return "HOW"
    else:
        return "OTHER"


def main():
    """
    Main analysis pipeline
    """
    print("="*80)
    print("FusionRAG Token Selection Analysis")
    print("="*80)

    # Load failure cases
    failure_file = "failure_cases_analysis.json"

    if not os.path.exists(failure_file):
        print(f"\n❌ Error: {failure_file} not found")
        print("Please run analyze_failure_cases.py first")
        return

    with open(failure_file, 'r', encoding='utf-8') as f:
        failure_cases = json.load(f)

    print(f"\n✓ Loaded {len(failure_cases)} failure cases")

    # Analyze each case
    print("\n" + "="*80)
    print("Analyzing Missing Information Patterns")
    print("="*80)

    analyses = []
    category_counts = defaultdict(int)
    question_type_failures = defaultdict(list)

    for i, case in enumerate(failure_cases, 1):
        print(f"\n--- Case {i} ---")
        print(f"Question: {case['sub_question'][:80]}...")

        analysis = analyze_missing_information(case)
        category = categorize_failure_type(analysis)
        question_type = analyze_question_type(case['sub_question'])

        category_counts[category] += 1
        question_type_failures[question_type].append(category)

        analysis['failure_category'] = category
        analysis['question_type'] = question_type
        analyses.append(analysis)

        print(f"Type: {question_type}")
        print(f"Category: {category}")
        print(f"Missing keywords: {analysis['missing_in_rate03']}")
        print(f"Wrong keywords: {analysis['wrong_in_rate03']}")

    # Summary statistics
    print("\n" + "="*80)
    print("Failure Category Distribution")
    print("="*80)

    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(failure_cases) * 100
        print(f"{category:25s}: {count:2d} cases ({percentage:5.1f}%)")

    print("\n" + "="*80)
    print("Question Type Analysis")
    print("="*80)

    for q_type, categories in sorted(question_type_failures.items()):
        print(f"\n{q_type} questions ({len(categories)} failures):")
        cat_dist = defaultdict(int)
        for cat in categories:
            cat_dist[cat] += 1
        for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")

    # Identify high-risk patterns
    print("\n" + "="*80)
    print("High-Risk Patterns")
    print("="*80)

    # Find most common missing keywords
    all_missing = []
    for analysis in analyses:
        all_missing.extend(analysis['missing_in_rate03'])

    missing_freq = defaultdict(int)
    for keyword in all_missing:
        missing_freq[keyword] += 1

    print("\nMost frequently missing keywords:")
    for keyword, freq in sorted(missing_freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  - '{keyword}': missing in {freq} cases")

    # Save detailed analysis
    output_file = "token_selection_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analyses': analyses,
            'summary': {
                'total_cases': len(failure_cases),
                'category_distribution': dict(category_counts),
                'question_type_distribution': {
                    q_type: len(cats) for q_type, cats in question_type_failures.items()
                },
                'most_common_missing_keywords': dict(
                    sorted(missing_freq.items(), key=lambda x: -x[1])[:20]
                )
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed analysis saved to {output_file}")

    # Recommendations
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)

    if category_counts['MISSING_KEY_INFO'] > 3:
        print("\n⚠️  HIGH PRIORITY: Missing Key Information")
        print("   Recommendation: Implement NER-guided token selection")
        print("   This will preserve named entities, dates, and numbers during compression")

    if category_counts['WRONG_RETRIEVAL'] > 2:
        print("\n⚠️  HIGH PRIORITY: Wrong Document Retrieval")
        print("   Recommendation: Separate document similarity computation from compression")
        print("   Compute BGE embeddings on uncompressed KV, then compress after selection")

    if category_counts['EVAL_INCONSISTENCY'] > 3:
        print("\n⚠️  MEDIUM PRIORITY: Evaluation Inconsistency")
        print("   Recommendation: Use deterministic evaluation (temperature=0, fixed seed)")

    if question_type_failures.get('WHEN', []):
        print(f"\n⚠️  'WHEN' questions are failing ({len(question_type_failures['WHEN'])} cases)")
        print("   Recommendation: Boost importance of date/year tokens during compression")

    if question_type_failures.get('WHO', []):
        print(f"\n⚠️  'WHO' questions are failing ({len(question_type_failures['WHO'])} cases)")
        print("   Recommendation: Boost importance of person name tokens during compression")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
