dataset_name='COFFEE-editor-correct-wrong'
dataset_format="input_output_codegen-only"

sh /home/lune/nas2/Projects/code_editing/qlora/scripts/finetune_codellama_7B.sh $dataset_name $dataset_format