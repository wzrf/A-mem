#!/bin/bash

PYTHON=/mnt/data/xmy/A-mem/.venv/bin/python3.10
SCRIPT=/mnt/data/xmy/A-mem/test_advanced_robust.py

export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH=./FusionRAG
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=sk-11ce7640e46049a6977c0d96ba855ffb

for draft_model in "qwen2.5-3B"; do ## qwen2.5-3B qwen2.5-1.5B
    for retrieve_k in 5; do ## 10 5
        for recomputation_rate in 1.0; do
            for use_rag in True False; do
                echo "========================================"
                echo "draft_model=${draft_model}, recomputation_rate=${recomputation_rate}, retrieve_k=${retrieve_k}"
                echo "========================================"

                $PYTHON $SCRIPT \
                    --skip_build true \
                    --recomputation_rate $recomputation_rate \
                    --draft_model $draft_model \
                    --qa_ratio 0.2 \
                    --ratio 1.0 \
                    --use_fusion_rag True \
                    --retrieve_k $retrieve_k \
                    --use_rag $use_rag

                echo ""
                echo "Finished: draft_model=${draft_model}, recomputation_rate=${recomputation_rate}, retrieve_k=${retrieve_k}"
                echo ""
            done
        done
    done
done