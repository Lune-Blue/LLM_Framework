# LLM Training & Inference Framework

This repository provides a modular framework for training and inference of Large Language Models (LLMs), integrating various state-of-the-art techniques and toolkits.  
It supports full fine-tuning, parameter-efficient training (e.g., QLoRA), and scalable inference using high-performance engines.

## 🏗️ Components Overview

### [`axolotl`](https://github.com/axolotl-ai-cloud/axolotl)
- Wrapper around HuggingFace's Trainer with structured config YAMLs.
- Supports LoRA, QLoRA, DeepSpeed, and more.
- Recommended for **scalable and reproducible training pipelines**.


### `dpo`
- Implements Direct Preference Optimization for **alignment with human preferences**.
- Requires a reward model and preference datasets.

### `ppo`
- Implements PPO for **reinforcement learning with human feedback (RLHF)**.
- Useful for controlling model behavior post-SFT.

### `qlora`
- Lightweight fine-tuning using QLoRA (quantized adapters).
- Optimized for **low-resource hardware** (e.g., single A100, consumer GPUs).

### `train`
- Custom and experimental training scripts.
- Good starting point for **prototyping** and model-specific tweaks.

### [`text-generation-inference`](https://github.com/huggingface/text-generation-inference)
- Meta’s TGI: High-performance inference server for LLMs.
- Supports batching, streaming, quantized models.

### [`vllm_inference`](https://github.com/vllm-project/vllm)
- vLLM-based inference engine for **fast and memory-efficient generation**.
- Ideal for serving large models with **continuous batching and KV caching**.

---

## ⚙️ Use Cases

- 🔬 Research on alignment methods (DPO, PPO)
- ⚡ Efficient model deployment (TGI, vLLM)
- 🧪 Prototyping training strategies (train/)
- 🛠️ Full-stack fine-tuning workflows (axolotl, qlora)


## 📄 License

This repository is for research and development purposes. Licensing may vary per submodule—refer to each folder for specific details.