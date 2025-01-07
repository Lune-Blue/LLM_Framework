dataset_name=$1
dataset_format='feedback-only'

CUDA_VISIBLE_DEVICES=0,1,2,3 sh qlora/scripts/finetune_codellama_split.sh $dataset_name $dataset_format