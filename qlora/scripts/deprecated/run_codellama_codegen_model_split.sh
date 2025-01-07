dataset_name=$1
dataset_format='codegen-only'

CUDA_VISIBLE_DEVICES=4,5,6,7 sh qlora/scripts/finetune_codellama_split.sh $dataset_name $dataset_format