# base_model='bigcode/octocoder'
base_model='codellama/CodeLlama-7b-Instruct-hf'
checkpoint_dir='/home/lune/nas2/Projects/code_editing/qlora/output/codellama-2-new-no-feedback-6k/checkpoint-1000'
merge_model_save_dir='/home/lune/nas2/Projects/code_editing/qlora/new_checkpoint/codellama-2-new-no-feedback-6k-1000'
cuda_visible_devices=0,1,2,3,4,5,6,7
tensor_parallel_size=8
prompt_key='without_feedback'
save_dir='/home/lune/nas2/Projects/code_editing/vllm/new_result/codellama-2-new-no-feedback-6k-1000'
sh /home/lune/nas2/Projects/code_editing/qlora/merge.sh $base_model $checkpoint_dir $merge_model_save_dir
echo "merge done"
sh /home/lune/nas2/Projects/code_editing/vllm/run_new.sh $cuda_visible_devices $merge_model_save_dir $tensor_parallel_size $prompt_key $save_dir
