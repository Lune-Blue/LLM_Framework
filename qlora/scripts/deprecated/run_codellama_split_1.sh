dataset_name='code-50k-final'
dataset_format='direct_code_edit'

CUDA_VISIBLE_DEVICES=0,1,2,3 sh /home/lune/nas2/Projects/code_editing/qlora/scripts/finetune_codellama_split.sh $dataset_name $dataset_format