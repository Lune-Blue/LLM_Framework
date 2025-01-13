model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -e CURL_CA_BUNDLE="" -e all_proxy=socks5://127.0.0.1:7890 \
    -e https_proxy=http://127.0.0.1:7890 -e http_proxy=http://127.0.0.1:7890 \
    -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model