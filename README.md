# RL Forgets! Towards Continual Policy Optimization

<p align="center">
  <a href="https://huggingface.co/datasets/MaolinLuo/MRCL"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MRCL%20Dataset-FFD21E.svg" alt="Hugging Face Dataset"></a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2607.04364"><img src="https://img.shields.io/badge/arXiv-2607.04364-B31B1B.svg" alt="arXiv"></a>
</p>

This repository contains the implementation of **Continual Policy Optimization (CPO)** and the **Multimodal Reasoning Continual Learning (MRCL)** benchmark. MRCL evaluates continual post-training on five recent and diverse tasks. CPO is a replay-free method that constrains movement on parameters identified as important to previous tasks.

Experiments on Qwen3-VL-2B, Qwen3-VL-4B, and Qwen3-VL-8B consistently show that CPO improves final retention over standard RL baselines while maintaining strong task adaptation, demonstrating that the method scales across model sizes.

![Examples from the five MRCL tasks](assets/mrcl-examples.png)

<details>
<summary><strong>Qwen3-VL-4B results</strong></summary>

![MRCL continual-learning results on Qwen3-VL-4B](assets/qwen3-vl-4b-mrcl.png)

![External capability results on Qwen3-VL-4B](assets/qwen3-vl-4b-ood.png)

</details>

## Setup

The paper experiments use Linux, NVIDIA H100 GPUs, Python 3.12, and CUDA 12.8. RL launchers are configured for eight GPUs; restrict visible devices with `CUDA_VISIBLE_DEVICES` if needed.

Training:

```bash
conda create -n trlQwen python=3.12 -y
conda activate trlQwen
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install psutil ninja einops
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
python -c "import nltk, sys, os; nltk_dir = os.path.join(sys.prefix, 'share', 'nltk_data'); os.makedirs(nltk_dir, exist_ok=True); nltk.download('punkt_tab', download_dir=nltk_dir)"
pip install deepspeed==0.16.4
```

Evaluation:

```bash
conda create -n vllmQwen python=3.12 -y
conda activate vllmQwen
pip3 install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_eval.txt
python -c "import nltk, sys, os; nltk_dir = os.path.join(sys.prefix, 'share', 'nltk_data'); os.makedirs(nltk_dir, exist_ok=True); nltk.download('punkt_tab', download_dir=nltk_dir)"
pip install vllm==0.11.2
```

The GPU configurations used in the paper are:

| Model | LoRA-based methods | Full SFT | RL methods |
| --- | ---: | ---: | ---: |
| Qwen3-VL-2B | 1 | 2 | 8 |
| Qwen3-VL-4B | 1 | 2 | 8 |
| Qwen3-VL-8B | 2 | 4 | 8 |

All numbers refer to NVIDIA H100 GPUs. RL methods include GRPO, GSPO, and CPO.

## Data

Set `BASE_PATH` in the launch script to the MRCL root:

```text
MRCL/
├── MedBookVQA/
├── Navigation/
├── We-Math2/
├── Puzzle/
└── FinMME/

<TASK>/
├── images/
└── jsons/
    ├── train/data.json
    └── test/data.json
```

Each `data.json` is a JSON array. A record contains `conversations` and an `image` field; FinMME numerical records also contain `tolerance`.

<details>
<summary><strong>External evaluation data</strong></summary>

Download [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/main), [MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/testmini.parquet), [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision/blob/main/data/test-00000-of-00001-3532b8d3f1b4047a.parquet), [MathVista](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet), [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA/tree/main/data), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar/blob/main/mmstar.parquet), [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA/tree/main/DocVQA), [CharXiv](https://huggingface.co/datasets/princeton-nlp/CharXiv/blob/main/val.parquet), [CountBenchQA](https://huggingface.co/datasets/vikhyatk/CountBenchQA/blob/main/data/test-00000-of-00001.parquet), and [POPE](https://github.com/AoiDragon/POPE/tree/main/output/coco).

```text
Eval_OOD_Datasets/
├── MMMU_pro/
│   ├── standard (4 options)/
│   ├── standard (10 options)/
│   └── vision/
├── POPE/
│   ├── coco_pope.json
│   └── coco/image/
├── MathVerse/
├── MathVision/
├── MathVista/
├── RealworldQA/
├── MMStar/
├── DocVQA/
├── Charxiv/
└── CountBenchQA/
```

#### Note:

- For MathVerse: Download the `testmini.parquet` and rename it as `test-00000-of-00001.parquet`.
- For MathVision and MathVista: Rename the file to `test-00000-of-00001.parquet`.
- For MMStar: Rename the file to `test-00000-of-00001.parquet` (original filename: `mmstar.parquet`).
- For POPE: Download the `coco_pope_adversarial.json` and rename it as `coco_pope.json`. Download and extract the [COCO](https://www.modelscope.cn/datasets/OmniData/COCO_2014/file/view/master/raw%2Fval2014.zip?id=28705&status=2) image files to the path `/POPE/coco/`.
- For DocVQA: Download the `validation-00000-of-00006.parquet` to `validation-00005-of-00006.parquet`.
- For Charxiv: Download the `val.parquet` and rename it as `validation-00000-of-00001.parquet`.

</details>

## Training and evaluation

Run commands from the repository root. First edit `BASE_MODEL` and `BASE_PATH` of the selected launcher.

```bash
# Qwen3-VL-2B: CPO
bash scripts/GSPO-CL/run_cpo.sh

# Qwen3-VL-4B: CPO
bash scripts/GSPO-CL-4B/run_cpo.sh

# Qwen3-VL-8B: CPO
bash scripts/GSPO-CL-8B/run_cpo.sh
```

<details>
<summary><strong>Baseline launchers</strong></summary>

```bash
# Qwen3-VL-2B: Full SFT
bash scripts/SFT-CL/run_all_FFT.sh
# Qwen3-VL-2B: LoRA
bash scripts/LoRA-CL/run_all.sh
# Qwen3-VL-2B: O-LoRA
bash scripts/OLoRA-CL/run_all.sh
# Qwen3-VL-2B: RegLoRA
bash scripts/RegLoRA-CL/run_all.sh
# Qwen3-VL-2B: KeepLoRA
bash scripts/KeepLoRA-CL/run_all.sh
# Qwen3-VL-2B: GRPO
bash scripts/GRPO-CL/run_cl.sh
# Qwen3-VL-2B: GSPO
bash scripts/GSPO-CL/run_cl.sh

# Qwen3-VL-4B: Full SFT
bash scripts/SFT-CL-4B/run_all_FFT.sh
# Qwen3-VL-4B: LoRA
bash scripts/LoRA-CL-4B/run_all.sh
# Qwen3-VL-4B: O-LoRA
bash scripts/OLoRA-CL-4B/run_all.sh
# Qwen3-VL-4B: RegLoRA
bash scripts/RegLoRA-CL-4B/run_all.sh
# Qwen3-VL-4B: KeepLoRA
bash scripts/KeepLoRA-CL-4B/run_all.sh
# Qwen3-VL-4B: GRPO
bash scripts/GRPO-CL-4B/run_cl.sh
# Qwen3-VL-4B: GSPO
bash scripts/GSPO-CL-4B/run_cl.sh

# Qwen3-VL-8B: Full SFT
bash scripts/SFT-CL-8B/run_all_FFT.sh
# Qwen3-VL-8B: LoRA
bash scripts/LoRA-CL-8B/run_all.sh
# Qwen3-VL-8B: O-LoRA
bash scripts/OLoRA-CL-8B/run_all.sh
# Qwen3-VL-8B: RegLoRA
bash scripts/RegLoRA-CL-8B/run_all.sh
# Qwen3-VL-8B: KeepLoRA
bash scripts/KeepLoRA-CL-8B/run_all.sh
# Qwen3-VL-8B: GRPO
bash scripts/GRPO-CL-8B/run_cl.sh
# Qwen3-VL-8B: GSPO
bash scripts/GSPO-CL-8B/run_cl.sh
```

</details>

For standalone zero-shot or external evaluation, edit the model and data paths in the corresponding script:

```bash
bash scripts/eval_zeroshot.sh
bash scripts/eval_zeroshot_ood.sh
```

## Citation

```bibtex
@article{luo2026rl,
    title={RL Forgets! Towards Continual Policy Optimization},
    author={Luo, Mao-Lin and Wang, Zhe-Xu and Zhou, Zi-Hao and Ye, Bo and Zhao, Jian and Zhang, Min-Ling and Wei, Tong},
    journal={arXiv preprint arXiv:2607.04364},
    year={2026}
}
```
