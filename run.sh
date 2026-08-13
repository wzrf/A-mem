#!/bin/bash

PYTHON=/mnt/data/xmy/A-mem/.venv/bin/python3.10
SCRIPT=/mnt/data/xmy/A-mem/test_advanced_robust.py

export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH=./FusionRAG
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=sk-11ce7640e46049a6977c0d96ba855ffb

for retrieve_k in 10 5; do
    for recomputation_rate in 0.3 1.0; do
        echo "========================================"
        echo "recomputation_rate=${recomputation_rate}, retrieve_k=${retrieve_k}"
        echo "========================================"

        $PYTHON $SCRIPT \
            --skip_build true \
            --recomputation_rate $recomputation_rate \
            --qa_ratio 0.1 \
            --ratio 0.2 \
            --use_fusion_rag True \
            --retrieve_k $retrieve_k

        echo ""
        echo "Finished: recomputation_rate=${recomputation_rate}, retrieve_k=${retrieve_k}"
        echo ""
    done
done