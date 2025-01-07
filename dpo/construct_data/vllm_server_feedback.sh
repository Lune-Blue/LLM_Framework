#!/bin/bash

A=0,1,2,3,4,5,6,7
B=8
CUDA_VISIBLE_DEVICES=$A
MODEL_NAME="/convei_nas2/hjchae/dongjin/code-editing/qlora/output/codellama-2-7b-only-feedback-6k/checkpoint-150-merged"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME --tensor-parallel-size $B --seed 42
