<p align="center">
    <img src="docs/assets/img/d2f/logo_lr.png" width="300">
</p>

## Discrete Diffusion Forcing (D2F): dLLMs Can Do Faster-Than-AR Inference

<p align="center">
  <a href="https://arxiv.org/abs/2508.09192"><b>📄 Paper</b></a> •
  <a href="https://SJTU-DENG-Lab.github.io/Discrete-Diffusion-Forcing/"><b>📝 Blog Post</b></a> •
  <a href="https://huggingface.co/spaces/zhijie3/D2F-LLaDA-Instruct-8B"><b>🚀 Online Demo</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"><b>🤗 D2F-Dream LoRA</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora"><b>🤗 D2F-LLaDA LoRA</b></a> 
</p>

<p align="center">
  <a href="https://discord.gg/aDWgxT6S2q"><b>💬 Discord</b></a> •
  <a href="docs/assets/img/d2f/wechat.png"><b>💬 Wechat</b></a>
</p>



https://github.com/user-attachments/assets/d9de6450-68d6-4caf-85c2-c7f384395c42


<p align="center">
  <br>
  <small><b>Real-time generation demo:</b> our D2F model (left) uses parallel block decoding, while the AR baseline (right) generates tokens sequentially. This visualizes the source of D2F's significant throughput advantage.</small>
</p>

<hr>

<p align="center">
    <img src="docs/assets/img/d2f/fig1_main_result.png" width="800">
    <br>
    <small><b>Inference throughput comparison:</b> D2F dLLMs surpass similarly-sized AR models in inference speed for the first time, achieving up to a <b>2.5x speedup</b> over LLaMA3 and a <b>>50x speedup</b> over vanilla dLLM baselines (Speed tests conducted on NVIDIA A100-PCIe-40GB GPUs).</small>
</p>

**Discrete Diffusion Forcing (D2F)** is a novel training and inference paradigm that, for the first time, enables open-source Diffusion Language Models (dLLMs) to surpass their autoregressive (AR) counterparts in inference speed. By introducing a highly efficient AR-diffusion hybrid model, D2F achieves:
- Up to a **2.5x speedup** over leading AR models like LLaMA3-8B.
- A staggering **50x+ acceleration** over vanilla dLLM baselines.
- Comparable generation quality on standard reasoning and coding benchmarks.
- **Integration with vLLM** to unlock the next tier of extreme inference acceleration.

This repository provides the code to reproduce our evaluation results and run generation demos.

## 🔥 News!
* Aug 20, 2025: We've released the training pipeline of D2F!
* Aug 8, 2025: We've released the inference code of D2F!
## Contents
- [🤔 How It Works](#-how-it-works)
- [📊 Performance Highlights](#-performance-highlights)
- [⚡️ Extreme Acceleration with vLLM Integration](#️-extreme-acceleration-with-vllm-integration)
- [🚀 Usage Guide](#-usage-guide)
- [🙏 Acknowledgements](#-acknowledgements)
- [©️ Citation](#️-citation)

## 🤔 How It Works

D2F overcomes the historical speed bottlenecks of dLLMs (KV Cache incompatibility and strict sequential dependencies) by restructuring the generation process.

**1. Hybrid Architecture:** D2F employs a **block-wise causal attention** mechanism. Attention *within* a block is bidirectional, preserving rich local context, while attention *between* blocks is causal. This simple but powerful change makes the model fully compatible with the standard KV Cache, drastically reducing redundant computations.

**2. Efficient Training via Asymmetric Distillation:** Instead of training from scratch, we distill a powerful, pre-trained bidirectional dLLM (teacher) into our cache-friendly D2F model (student). The student learns to match the teacher's output with only a limited, causal view of the context.

<p align="center">
    <img src="docs/assets/img/d2f/fig3_overview.png" width="800">
    <br>
    <small><b>Overview of Discrete Diffusion Forcing (D2F):</b> A D2F model (student) with a KV-cache-friendly block-wise causal attention mask is trained to mimic a powerful, pre-trained bidirectional dLLM (teacher), efficiently inheriting its capabilities.</small>
</p>

**3. High-Throughput Pipelined Decoding:** D2F is trained to predict future blocks based on *partially incomplete* prefixes. This enables a **pipelined parallel decoding** algorithm during inference, where multiple blocks are refined simultaneously in an asynchronous workflow, maximizing GPU utilization and throughput.

<p align="center">
    <img src="docs/assets/img/d2f/fig4_pipeline.png" width="800">
    <br>
    <small><b>Visualization of our pipelined parallel decoding:</b> New blocks are dynamically added and decoded in parallel with their predecessors, moving from a conservative "semi-activated" state to an aggressive "fully-activated" state. This creates a continuous, high-throughput generation flow.</small>
</p>

https://github.com/user-attachments/assets/41a0176b-e4ae-4f8b-95a6-daed7af2a027

<p align="center">
  <br>
  <small><b>A slow-motion demonstration of the parallel decoding process within a single block of D2F. Watch as multiple tokens within the block are refined simultaneously, showcasing the efficiency of our approach.</small>
</p>

## 📊 Performance Highlights

D2F delivers transformative speedups while maintaining or improving scores. Below is a comprehensive summary of performance on **LLaDA-Instruct-8B** and **Dream-Base-7B**, comparing our method against the original baseline and the previous SOTA acceleration method, Fast-dLLM.

<center>

**Performance on LLaDA-Instruct-8B**
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Benchmark</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
      <th style="padding: 8px; border: 1px solid #ddd;">LLaDA-Instruct (Baseline)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (SOTA)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">D2F-LLaDA (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>GSM8K-4-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">7.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>52.5 <font color="green">(7.3x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>78.9</b></td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.3</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>MBPP-3-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">0.9</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">15.3</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>47.6 <font color="green">(52.9x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><b>39.0</b></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">36.4</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">38.0</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>HumanEval-0-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">2.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;">19.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>81.6 <font color="green">(29.1x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">36.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.4</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>40.2</b></td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>Math-4-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">21.1</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">42.5</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>90.2 <font color="green">(4.3x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">23.7</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">22.4</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><b>29.1</b></td>
    </tr>
  </tbody>
</table>

**Performance on Dream-Base-7B**
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Benchmark</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Dream-Base (Baseline)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (SOTA)</th>
      <th style="padding: 8px; border: 1px solid #ddd;">D2F-Dream (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>GSM8K-CoT-8-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">9.5</td>
      <td style="padding: 8px; border: 1px solid #ddd;">49.8</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>91.2 <font color="green">(9.6x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">75.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;">75.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>77.6</b></td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>MBPP-3-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">10.4</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">73.2</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>105 <font color="green">(10.1x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">56.2</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">51.0</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><b>56.4</b></td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle;"><strong>HumanEval-0-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">20.2</td>
      <td style="padding: 8px; border: 1px solid #ddd;">60.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>73.2 <font color="green">(3.6x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
      <td style="padding: 8px; border: 1px solid #ddd;">53.0</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>55.5</b></td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border: 1px solid #ddd; vertical-align: middle; background-color: #fafafa;"><strong>Math-4-shot</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">TPS ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">9.9</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">67.0</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><strong>98.8 <font color="green">(10.0x)</font></strong></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">Score ↑</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">35.8</td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;"><b>37.6</b></td>
      <td style="padding: 8px; border: 1px solid #ddd; background-color: #fafafa;">35.4</td>
    </tr>
  </tbody>
</table>
</center>

## ⚡️ Extreme Acceleration with vLLM Integration

To push the boundaries of inference speed, we've integrated D2F with a **preliminary vLLM-based engine**. This unlocks a multiplicative speedup on top of our already-accelerated model, showcasing the immense potential for production environments.

<center>

<strong>HumanEval-0-shot with vLLM</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Model</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Dream-Base (Baseline)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">20.2 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">D2F-Dream (Ours)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.2 <font color="green">(3.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
    </tr>
    <tr style="background-color:#E8F5E9;">
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-Dream + vLLM (Ours)</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>131.7 <font color="green">(6.5x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">40.2</td>
    </tr>
  </tbody>
</table>
<br>
<small>Our D2F-Dream model with a preliminary vLLM engine achieves a <b>6.5x speedup</b> over the original Dream-Base, though we observe a score drop that we are actively working to resolve through optimized kernels.</small>

</center>

> **Implementation Notes:**
> The current vLLM integration is an initial proof-of-concept. It already provides a significant performance boost by leveraging Flex Attention, but there is substantial room for further optimization. Our future work will focus on implementing specialized CUDA kernels and other advanced vLLM features to maximize speed while restoring the score.

## 🚀 Usage Guide

### 1. Installation

First, clone the repository and set up the environment.

```shell
# Clone the repository
git clone https://github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing.git
cd Discrete-Diffusion-Forcing
```

#### Environment Configuration

##### UV (Recommended)

```shell
uv sync
```

##### Conda 

```shell
# Create and activate a conda environment
conda create -n d2f python=3.10
conda activate d2f

# Install dependencies
pip install -r requirements.txt
```

#### vLLM Installation

vLLM is comming soon, right now we only implemented the basic functions of vLLM.

### 2. Evaluation
All evaluation scripts are located in the `D2F-eval` directory.

```shell
cd D2F-eval
```

To evaluate the **D2F-Dream** model on all benchmarks, run:

```shell
shell eval_dream.sh
```

To evaluate the **D2F-LLaDA** model on all benchmarks, run:

```shell
shell eval_llada.sh
```
The results will be saved in the `output_path` specified within the shell scripts.

> ### ❗️ Important Notice for HumanEval
> The `HumanEval` benchmark requires a post-processing step to sanitize the generated code and calculate the final `pass@1` score. After the evaluation script finishes, run the following command:
> ```shell
> python postprocess_code.py {path/to/your/samples_humaneval_xxx.jsonl}
> ```
> Replace the path with the actual path to your generated samples file, which can be found in the specified `output_path`.

### 3. Training
All training scripts and configurations are located in the `D2F-train` directory.
```shell
# Navigate to the training directory
cd D2F-train
```
Before starting the training, you need to configure the paths for your dataset, models, and output directories. Modify the relevant paths in the configuration files located inside the `config` folder.

Once the configuration is set, you can start the training process by running:
```shell
bash train.sh
```

### 4. Generation Demo

We provide simple scripts to demonstrate the generation process and compare D2F with a standard AR baseline.
```shell
# To run a demo with the D2F pipelined block generation method:
python generate_llada_demo_block.py

# To compare, run a demo with the baseline AR generation method:
python generate_llada_demo_ar.py
```
You can inspect these files to see how to use the D2F model for inference in your own projects.

## 📚 Future Works

- [x] Implement dLLM-suported vLLM (preliminary).
- [ ] Implement fused dLLM-specific decoding kernels for vLLM to maximize performance and restore scores.
- [ ] Implement distributed inference with multi-GPUs in vLLM.
- [ ] Implement CUDA graph capturing for dynamic sequences in vLLM.

## 🙏 Acknowledgements
Our work builds upon the foundations laid by the original **LLaDA** and **Dream** models. We thank their authors for making their work public. We are also grateful for the powerful open-source tools from Hugging Face and the vLLM team that made this research possible.

## ©️ Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@article{wang2025diffusion,
  title={Diffusion llms can do faster-than-ar inference via discrete diffusion forcing},
  author={Wang, Xu and Xu, Chenkai and Jin, Yijie and Jin, Jiachun and Zhang, Hao and Deng, Zhijie},
  journal={arXiv preprint arXiv:2508.09192},
  year={2025}
}


```
