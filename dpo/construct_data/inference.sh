#!/bin/bash

split="eval"

python -m dpo.construct_data.run_local_api_models \
    --dataset_name DLI-Lab/new-code-feedback-final-6k-data \
    --split $split \
    --inference_model_name /convei_nas2/hjchae/dongjin/code-editing/qlora/output/codellama-2-7b-only-feedback-6k/checkpoint-150-merged \
    --inference_model_port 8000 \
    --save_dir dpo/data/${split}_inference 