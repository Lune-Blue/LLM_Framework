#!/bin/bash

function start_two_servers() {
    # Feedback Server
    touch ${FEEDBACK_SERVER_LOG_FILE}
    touch ${GENERATE_SERVER_LOG_FILE}
    echo "Starting feedback server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_FEEDBACK python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_FEEDBACK \
    --tensor-parallel-size $TENSOR_SIZE_TWO \
    --seed 42 \
    --port $PORT_FEEDBACK > ${FEEDBACK_SERVER_LOG_FILE} 2>&1 &
    FEEDBACK_SERVER_PID=$!
    echo "Feedback server PID: ${FEEDBACK_SERVER_PID}"

    # Generate Server
    echo "Starting generate server..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_GENERATE python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH_GENERATE \
    --tensor-parallel-size $TENSOR_SIZE_TWO \
    --seed 42 \
    --port $PORT_GENERATE > ${GENERATE_SERVER_LOG_FILE} 2>&1 &
    GENERATE_SERVER_PID=$!
    echo "Generate server PID: ${GENERATE_SERVER_PID}"

    # Wait for Feedback Server
    echo "Waiting for feedback server to start..."
    while ! grep "Uvicorn running on" ${FEEDBACK_SERVER_LOG_FILE}; do
        sleep 10
    done

    # Wait for Generate Server
    echo "Waiting for generate server to start..."
    while ! grep "Uvicorn running on" ${GENERATE_SERVER_LOG_FILE}; do
        sleep 10
    done

    echo "Both servers started successfully!"
}

function start_run_script_two() {
    echo "Starting the run script for both servers..."
    python inference.py \
    --feedback_model_name ${MODEL_PATH_FEEDBACK} \
    --feedback_model_port ${PORT_FEEDBACK} \
    --generation_model_name ${MODEL_PATH_GENERATE} \
    --generation_model_port ${PORT_GENERATE} \
    --save_dir ${SAVE_DIR}
    # sh ${RUN_SCRIPT} ${MODEL_PATH_FEEDBACK} ${PORT_FEEDBACK} ${MODEL_PATH_GENERATE} ${PORT_GENERATE} ${PROMPT} ${SAVE_LOCATION} ${USE_FB} > ${RUN_SCRIPT_LOG_FILE} 2>&1 &
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
    # Ask user if they want to kill the server
    kill ${FEEDBACK_SERVER_PID}
    kill ${GENERATE_SERVER_PID}
    echo "Server process terminated."
    exit 0
}

FIRST_MODEL_NAME=$(basename $2)
LOG_DIR="logs"
trap handle_sigint SIGINT

if [ "$#" -ne 10 ]; then
    echo "Error: Incorrect number of arguments for two-server version."
    exit 1
fi
SECOND_MODEL_NAME=$(basename $6)

GENERATE_SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_generate_server.log"
FEEDBACK_SERVER_LOG_FILE="${LOG_DIR}/${FIRST_MODEL_NAME}_feedback_server.log"
CUDA_DEVICES_FEEDBACK=$1
MODEL_PATH_FEEDBACK=$2
PORT_FEEDBACK=$3
CUDA_DEVICES_GENERATE=$4
MODEL_PATH_GENERATE=$5
PORT_GENERATE=$6
TENSOR_SIZE_TWO=$7
SAVE_DIR=$8
PROMPT=${9}
USE_FB=${10}
echo "CUDA_DEVICES_FEEDBACK: ${CUDA_DEVICES_FEEDBACK}"
echo "MODEL_PATH_FEEDBACK: ${MODEL_PATH_FEEDBACK}"
echo "PORT_FEEDBACK: ${PORT_FEEDBACK}"
echo "CUDA_DEVICES_GENERATE: ${CUDA_DEVICES_GENERATE}"
echo "MODEL_PATH_GENERATE: ${MODEL_PATH_GENERATE}"
echo "PORT_GENERATE: ${PORT_GENERATE}"
echo "TENSOR_SIZE_TWO: ${TENSOR_SIZE_TWO}"
echo "PROMPT: ${PROMPT}"
echo "USE_FB: ${USE_FB}"

echo "Result will be saved in: ${SAVE_DIR}"

start_two_servers
start_run_script_two
kill ${FEEDBACK_SERVER_PID}
kill ${GENERATE_SERVER_PID}
exit 0