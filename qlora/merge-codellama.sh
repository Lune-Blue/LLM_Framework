# base_model_name_or_path="bigcode/octocoder"
base_model_name_or_path="codellama/CodeLlama-7b-Instruct-hf"
peft_checkpoint=$1
# peft_checkpoint="/home/lune/nas2/Projects/code_editing/qlora/new_output/codellama-2-new-feedback-6k/checkpoint-550"

# Parse the last name for base model
base_model_last_name=$(basename $base_model_name_or_path)

# Parse the desired folder name for peft_checkpoint
peft_checkpoint_directory=$(dirname $peft_checkpoint)
peft_checkpoint_last_name=$(basename $peft_checkpoint_directory)
peft_checkpoint_step=$(basename $peft_checkpoint)

model_name_to_be_saved="checkpoints/${base_model_last_name}/${peft_checkpoint_last_name}-${peft_checkpoint_step}"

echo "merge start"
echo "base model: $base_model_name_or_path"
echo "peft checkpoint: $peft_checkpoint"
echo "save location: $model_name_to_be_saved"
python qlora/merge-peft-adapters.py \
    --base_model_name_or_path $base_model_name_or_path \
    --peft_model_path $peft_checkpoint \
    --save_path $model_name_to_be_saved
