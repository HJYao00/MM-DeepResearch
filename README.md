<div align="center">

<h1> MM-DeepResearch: A Simple and Effective Multimodal Agentic Search Baseline </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.


<h5 align="center"> 

<a href='https://arxiv.org/abs/2603.01050'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/HuanjinYao/MM-DeepResearch-8B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://huggingface.co/datasets/HuanjinYao/MM-DeepResearch-corpus'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>


</h5>
</div>


## 🔔 News
- [x] **`Mar 13, 2026.`** We released the evaluation code and the [model](https://huggingface.co/HuanjinYao/MM-DeepResearch-8B). The training code will be open-sourced soon!
- [x] **`Mar 1, 2026.`** We released MM-DeepResearch and made the paper available on [arxiv](https://arxiv.org/abs/2603.01050).

## 🔍 Evaluation
### Step 1: Launch the deep research agent and the judge/summary model.

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

----------

Then launch the judge/summary model by [vLLM](https://docs.vllm.ai/en/latest/) or [SGLang](https://docs.sglang.io/index.html). We recommend [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) or [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) as the judge and summary model here. In general, larger models provide more reliable judgment and higher-quality summaries.
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --port 9000 \
    --tp-size 4 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 \
    --context-length 262144
```

### Step 2: Prepare the test dataset and search APIs for evaluation

**Dataset.**  
The test dataset format is the same as that used in VeRL. You can download the test dataset [here](https://huggingface.co/datasets/HuanjinYao/MM-DeepResearch-corpus/blob/main/MMSearch_test.parquet) or run the following script to generate the test dataset:

```bash
python3 eval/data_preprocess/preprocess_MMSearch.py
```

In addition, you need to prepare the search APIs required for evaluation. We currently support SerpAPI and Serper. And Jina Reader API is required
In addition, evaluation requires access to search APIs (support [SerpAPI](https://serpapi.com/) and [Serper](https://serper.dev/)), and [Jina Reader API](https://jina.ai/).

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
