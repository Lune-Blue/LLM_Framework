#!/bin/bash

function start_server(){
    echo "Starting server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_SINGLE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_SINGLE \
    --trust-remote-code \
    --tensor-parallel-size $TENSOR_SIZE_SINGLE \
    --seed 42 \
    --port $PORT_SINGLE > ${SERVER_LOG_FILE} 2>&1 &


    # sh ${RUN_SERVER} ${CUDA_DEVICES_SINGLE} ${MODEL_PATH_SINGLE} ${TENSOR_SIZE_SINGLE} ${PORT_SINGLE} > ${SERVER_LOG_FILE} 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    echo "Waiting for server to start..."
    while ! grep "Uvicorn running on" ${SERVER_LOG_FILE}; do
        sleep 10
    done
}

function start_run_script_single() {
    echo "Starting the run script..."
    python inference.py \
    --generation_model_name ${MODEL_PATH_SINGLE} \
    --generation_model_port ${PORT_SINGLE} \
    --prompt_key ${PROMPT} \
    --use_feedback ${USE_FB} \
    --save_dir ${SAVE_DIR}
    # \
    # > "${RUN_SCRIPT_LOG_FILE}" &

    RUN_SCRIPT_PID=$!
    # echo "Run script PID: ${RUN_SCRIPT_PID}"

    # echo "Waiting for run script to complete..."
    # while ! grep "Done!" ${RUN_SCRIPT_LOG_FILE}; do
    #     sleep 10
    # done
}

function handle_sigint() {
    echo -e "\nYou've stopped the main script. The run script process will be terminated."
    kill ${RUN_SCRIPT_PID} || true
    kill ${SERVER_PID}
    echo "Server process terminated."

    exit 0
}




LOG_DIR="logs"
trap handle_sigint SIGINT
FIRST_MODEL_NAME=$(basename $2)
if [ "$#" -ne 7 ]; then
    echo "Error: Incorrect number of arguments for single-server version."
    exit 1
fi

SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_server.log"
CUDA_DEVICES_SINGLE=$1
MODEL_PATH_SINGLE=$2
PORT_SINGLE=$3
TENSOR_SIZE_SINGLE=$4
SAVE_DIR=$5
PROMPT=${6}
USE_FB=${7}
echo "Result will be saved in: ${SAVE_DIR}"

start_server
start_run_script_single
echo "Saved in: ${SAVE_DIR}"
kill ${SERVER_PID}
exit 0