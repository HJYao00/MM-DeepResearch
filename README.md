<div align="center">

<h1> MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟. </h5>


<h5 align="center"> 

<a href='https://arxiv.org/abs/2603.01050'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/HuanjinYao/MM-DeepResearch-8B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/HuanjinYao/MM-DeepResearch-corpus'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'></a>
<a href='https://https://huggingface.co/HuanjinYao/MM-DeepResearch-8B'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a>


</h5>
</div>


## 📋 Table of Contents
- [🔔 News](#-news)
- [🛠️ Installation](#-installation)
- [🏋️ Training](#-training)
- [🔍 Evaluation](#-evaluation)
- [🔗 Citation](#-citation)
- [🙏 Acknowledgment](#-acknowledgment)


## 🔔 News
- [x] **`April 28, 2026.`** We released the training code!
- [x] **`Mar 13, 2026.`** We released the evaluation code and the [model](https://huggingface.co/HuanjinYao/MM-DeepResearch-8B). The training code will be open-sourced soon!
- [x] **`Mar 1, 2026.`** We released MM-DeepResearch and made the paper available on [arxiv](https://arxiv.org/abs/2603.01050).



## 🛠️ Installation
### Install Dependencies
```bash
cd verl
pip3 install .
pip3 install -r ./requirements_sglang.txt

pip3 install sglang[all]==0.5.5.post3
pip3 install qwen-vl-utils -U
pip3 install vllm==0.11.0

pip3 install flash-attn==2.8.3 --no-build-isolation
pip3 install faiss-gpu-cu12==1.8.0.0

pip3 install datasets==4.2.0
pip3 install google-search-results
```

## 🏋️ Training

MM-DeepResearch is trained with multi-turn agentic GRPO reinforcement learning built on [VeRL](https://github.com/TIGER-AI-Lab/VERL). During training, the model interacts with offline retrieval engines that serve as environment simulators, enabling the agent to learn when and how to invoke different search tools.

### Step 1: Prepare the training dataset and offline corpus

Download the training corpus from [HuggingFace](https://huggingface.co/datasets/HuanjinYao/MM-DeepResearch-corpus). The corpus includes three types of retrieval data:

| File | Purpose |
|------|---------|
| `lens_cached_data.jsonl` & `images.tar.gz` | Image-to-image search. Run the provided script to convert image paths to absolute paths. |
| `image_search_result_rag.parquet` & `jina-clip-v2_Flat_image.index` | Text-to-image search. |
| `merged_reindexed_new.jsonl` & `e5_Flat.index` | Text-to-text search. |
| 'training_data_rl.parquet'  & 'MMSearch_test.parquet' | Training and testing dataset. |

### Step 2: Launch the offline retrieval engines and judge model

Launch the offline retrieval servers for text-to-text and text-to-image search. These are FastAPI servers backed by FAISS GPU indices:

```bash
# Text-to-text retrieval (E5-based, default port 9000)
bash verl/run_scripts/launch_text_engine.sh

# Text-to-image retrieval (Jina CLIP-based, default port 9001)
bash verl/run_scripts/launch_mm_engine.sh
```

Then launch the judge model for reward computation. You may use any model you prefer; we recommend [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) or larger models for more reliable reward signals:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --port 8001 \
    --tp-size 4 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 \
    --context-length 262144
```

### Step 3: Start agentic RL training

Before running, update the following configurations in `verl/run_scripts/mm_deepresearch.sh` to match your local setup:

- `JUDGE_API_BASE`: Judge model API address (e.g., `http://your_ip:8001/v1`)
- `SEARCH_CACHED_DATA_PATHS`: Path to the `lens_cached_data.jsonl` file for image-to-image search
- `actor_rollout_ref.model.path`: Path to the base model (e.g., `Qwen3-VL-8B-Instruct`)
- `data.train_files` / `data.val_files`: Paths to training and validation data

Then start training:

```bash
bash run_scripts/mm_deepresearch.sh
```

The training uses GRPO with the following key hyperparameters (see the [full config](verl/examples/sglang_multiturn/config/search_multiturn_grpo.yaml) for details):

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | GRPO |
| Training batch size | 64 |
| Learning rate | 1e-6 |
| Rollout per prompt (n) | 5 |
| Max prompt length | 60,000 |
| Max response length | 5,000 |
| Max user/assistant turns | 4 / 4 |
| Total epochs | 35 |

## 🔍 Evaluation

### Step 1: Launch the deep research agent and the judge/summary model

Start the deep research agent first:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path HuanjinYao/MM-DeepResearch-8B \
    --port 8000  \
    --tp-size 2 \
    --host 0.0.0.0 \
    --context-length 262144 \
    --trust-remote-code
```

Then launch the judge/summary model by [vLLM](https://docs.vllm.ai/en/latest/) or [SGLang](https://docs.sglang.ai/index.html). We recommend [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) or [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) as the judge and summary model. In general, larger models provide more reliable judgment and higher-quality summaries.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --port 9000 \
    --tp-size 4 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 \
    --context-length 262144
```

### Step 2: Prepare the test dataset and search APIs

**Dataset.**  
The test dataset format is the same as that used in VeRL. You can download the test dataset [here](https://huggingface.co/datasets/HuanjinYao/MM-DeepResearch-corpus/blob/main/MMSearch_test.parquet) or run the following script to generate it:

```bash
python3 eval/data_preprocess/preprocess_MMSearch.py
```

**Search APIs.**  
Evaluation requires access to search APIs (we support [SerpAPI](https://serpapi.com/) and [Serper](https://serper.dev/)), and the [Jina Reader API](https://jina.ai/) for fetching and summarizing web page content.

### Step 3: Complete the code for image-to-image search

Since [image-to-image search](https://serpapi.com/google-lens-api) only supports searches using publicly accessible image URLs, you need to implement an image upload step [here](https://github.com/HJYao00/MM-DeepResearch/blob/main/eval/tool_image_search_lens.py#L163) that uploads local images to a public server and obtains public URLs for search.
> **Note:** We highly recommend uploading the input images in advance to avoid potential upload errors during evaluation.
> 
### Step 4: Run evaluation

Finally, you can start the evaluation with the following command:

```bash
cd eval
bash scripts/run_eval_mmsearch_search_mp.sh
```



## 🔗 Citation
If you find this repository is useful, please star🌟 this repo and cite🖇️ our paper.
```bibtex
@article{yao2026mm,
  title={MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline},
  author={Yao, Huanjin and Yin, Qixiang and Yang, Min and Zhao, Ziwang and Wang, Yibo and Luo, Haotian and Zhang, Jingyi and Huang, Jiaxing},
  journal={arXiv preprint arXiv:2603.01050},
  year={2026}
}
```


## 🙏 Acknowledgment
Our work is primarily based on the following codebases. We are sincerely grateful for their work.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Used for supervised fine-tuning of our base multimodal models.
- [VeRL](https://github.com/hiyouga/LLaMA-Factory): Used to perform multi-turn agentic reinforcement learning.
- [Search-R1](https://github.com/open-compass/VLMEvalKit): Our agentic search framework is inspired by the Search-R1 implementation.
