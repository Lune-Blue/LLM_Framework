NUM_GPUS=4
token=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
text-generation-launcher --model-id meta-llama/Llama-2-7b-hf \
    --sharded true --num-shard $NUM_GPUS --quantize bitsandbytes

# text-generation-launcher --model-id codellama/CodeLlama-13b-Instruct-hf \
#     --num-shard $NUM_GPUS --quantize bitsandbytes