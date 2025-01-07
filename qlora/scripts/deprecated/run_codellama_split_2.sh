dataset_name='code-6k-feedback-silver-refined'
dataset_format='feedback-only'

CUDA_VISIBLE_DEVICES=4,5,6,7 sh /home/lune/nas2/Projects/code_editing/qlora/scripts/finetune_codellama_split.sh $dataset_name $dataset_format