import yaml
import subprocess
import sys
import shutil
import os
import getpass
from datetime import datetime, timedelta


def current_kst():
    """Get the current time in KST (Korea Standard Time)."""
    UTC_OFFSET = 9
    utc_time = datetime.utcnow()
    kst_time = utc_time + timedelta(hours=UTC_OFFSET)
    return kst_time.strftime("%Y-%m-%d %H:%M:%S KST")


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference_with_yaml_config.py <path_to_yaml_config>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config = read_yaml(config_file_path)

    # Derive the save directory from the model name or path
    model_last_name = config["GENERATE_MODEL_NAME_OR_PATH"].split("/")[-1]
    use_two_servers = config["USE_TWO_SERVERS"]
    if use_two_servers == "yes":
        first_model_name = config["FEEDBACK_MODEL_NAME_OR_PATH"].split("/")[-1]
        second_model_name = model_last_name
        save_dir = f"generation_results/{first_model_name}-{second_model_name}"
        command = ["script/vllm_multi_inference.sh"]
        command.extend(
            [
                config["FEEDBACK_CUDA_DEVICES"],  
                config["FEEDBACK_MODEL_NAME_OR_PATH"], 
                str(config["FEEDBACK_PORT"]), 
            ]
        )
    else:
        save_dir = f"generation_results/{model_last_name}-single-model"
        command = ['script/vllm_single_inference.sh']
    command.extend(
        [
            config["GENERATE_CUDA_DEVICES"],  
            config["GENERATE_MODEL_NAME_OR_PATH"],
            str(config["GENERATE_PORT"]),
            str(config["TENSOR_PARALLEL_SIZE"]),
            save_dir,
            config["prompt_key"],
            config["use_feedback"],
        ]
    )
    # Add execution details to the configuration
    config["execution_details"] = {"timestamp_kst": current_kst(), "user": getpass.getuser(), "cwd": os.getcwd()}

    # Save the modified configuration to the output directory
    output_config_path = os.path.join(save_dir, os.path.basename(config_file_path))
    output_dir = os.path.dirname(output_config_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_yaml(config, output_config_path)


    for i, c in enumerate(command):
        print(i, c)

    # print(command)
    print(f"Num arguments :{len(command)-1}")
    subprocess.run(command)
