dataset_name='DLI-Lab/code-selector-2100-checkpoint'

CUDA_VISIBLE_DEVICES=4,5,6,7 sh /home/lune/nas2/Projects/code_editing/qlora/classification/finetune_codellama_classification.sh $dataset_name