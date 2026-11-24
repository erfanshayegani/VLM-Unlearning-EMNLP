# Can Textual Unlearning Solve Cross-Modality Safety Alignment?

Official code for the EMNLP 2024 (Findings) paper
**[Can Textual Unlearning Solve Cross-Modality Safety Alignment?](https://aclanthology.org/2024.findings-emnlp.574/)**
by Trishna Chakraborty\*, Erfan Shayegani\*, Zikui Cai, Nael Abu-Ghazaleh, M. Salman Asif, Yue Dong, Amit K. Roy-Chowdhury and Chengyu Song.

\* Co-first authors (equal contribution).

> **Content warning:** this repository deals with jailbreaks and harmful model outputs. Some prompts in the code are offensive.

## Overview

Vision-language models (VLMs) inherit their language model's safety alignment, but new
modalities (images) open new attack surfaces: cross-modality jailbreaks. We study whether
**unlearning on text only**, applied to the LLM inside the VLM, carries over to
multi-modal attacks. We compare it with **multi-modal unlearning** and **multi-modal
supervised fine-tuning (SFT)**.

The main finding, across seven datasets: textual unlearning lowers the attack success rate of
both text-based and vision-text attacks to below 8% (in some cases nearly 2%) while preserving
utility. Multi-modal unlearning gives no additional benefit and needs much more compute.

All methods train a LoRA adapter on the language model of a 4-bit (NF4) quantized LLaVA:

| Method | Script | Harmful (forget) data | Normal (retain) data |
|---|---|---|---|
| Textual unlearning | `unlearn_harm_VLM.py` | PKU-SafeRLHF (text) | TruthfulQA (text) |
| Multi-modal unlearning | `multimodal-unlearn_harm_VLM.py` | JailBreakV-28K or FigStep (image+text) | VQA-v2, 520 examples (image+text) |
| Multi-modal SFT | `multimodal-SFT_harm_VLM.py` | JailBreakV-28K or FigStep mapped to a refusal | VQA-v2, 520 examples (image+text) |
| Text-only LLM unlearning (reference) | `unlearn_harm.py` | PKU-SafeRLHF | TruthfulQA |

Unlearning loss = gradient ascent on harmful answers + KL divergence to the original model on
normal data (+ optional "random mismatch" toward a refusal for textual unlearning).

Supported models: `llava-hf/llava-1.5-7b-hf` and `llava-hf/llava-v1.6-mistral-7b-hf`.

## Repository structure

```
├── unlearn_harm_VLM.py              # textual unlearning on a VLM
├── multimodal-unlearn_harm_VLM.py   # multi-modal unlearning
├── multimodal-SFT_harm_VLM.py       # multi-modal SFT baseline
├── unlearn_harm.py                  # text-only LLM unlearning (e.g. Llama-2)
├── utils.py                         # text dataloaders & losses
├── multimodal_utils.py              # image+text dataloaders & losses
├── model_utils.py                   # LLaVA / LLaVA-NeXT loading, quantization, saving
├── paths.py                         # all data locations (overridable via env vars)
├── inference/                       # response generation on the 4 benchmarks
│   ├── PKU_RLHF/   ├── VQA/   ├── Figstep/   └── JBPieces/
├── evaluation/evaluate.py           # reward model (OpenAssistant / Skywork), Detoxify, Perspective API
├── scripts/train.sh                 # training commands
├── scripts/run_inference.sh         # run all benchmarks for one adapter
└── data/README.md                   # how to obtain every dataset
```

## Setup

```bash
conda create -n vlm-unlearn python=3.10 -y && conda activate vlm-unlearn
pip install -r requirements.txt
huggingface-cli login            # LLaVA weights / datasets
wget -P data/ https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv
```

> Pinned versions matter: recent `transformers` releases changed how LLaVA inserts image
> tokens. See `requirements.txt`.
>
> **VQA-v2** is loaded from [`lmms-lab/VQAv2`](https://huggingface.co/datasets/lmms-lab/VQAv2):
> the first 520 validation examples, read from the first parquet shard only (~90 MB), so the full
> dataset is not downloaded.

Then get the other datasets as described in **[data/README.md](data/README.md)**.

## Training

```bash
# Textual unlearning (public data only)
python unlearn_harm_VLM.py --use_lora --model_name llava-hf/llava-1.5-7b-hf \
    --bad_weight 0.05 --random_weight 1 --normal_weight 1 \
    --model_save_dir models/llava1.5-7b_text_unlearned --log_file logs/text_unlearn.log

# Multi-modal unlearning
python multimodal-unlearn_harm_VLM.py --use_lora --model_name llava-hf/llava-1.5-7b-hf \
    --harmful_data jailbreakv --bad_weight 0.05 --normal_weight 1 \
    --model_save_dir models/llava1.5-7b_multimodal_unlearned --log_file logs/mm_unlearn.log

# Multi-modal SFT
python multimodal-SFT_harm_VLM.py --use_lora --model_name llava-hf/llava-1.5-7b-hf \
    --harmful_data jailbreakv \
    --model_save_dir models/llava1.5-7b_multimodal_sft --log_file logs/mm_sft.log
```

`scripts/train.sh` lists all configurations. Add `--push_to_hub <your-hf-id>/<repo>` to also upload the adapter.
Hyper-parameters (LR 3e-4, batch size 2, AdaLoRA r=32 on the LLM's `q_proj`/`v_proj`, 1000 unlearning steps) are the script defaults.

## Inference & evaluation

```bash
# Responses of the adapter and of the original model on PKU-SafeRLHF, VQA-v2, FigStep, Jailbreak-in-Pieces
bash scripts/run_inference.sh models/llava1.5-7b_text_unlearned text_unlearn

# Score them
python evaluation/evaluate.py --csv inference/VQA/output_CSVs/text_unlearn_VQA.csv --metrics reward
# reward model: OpenAssistant/reward-model-deberta-v3-large-v2 by default; Skywork is also supported
python evaluation/evaluate.py --csv inference/VQA/output_CSVs/text_unlearn_VQA.csv --metrics reward \
    --reward_model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2   # needs transformers>=4.43 (separate env)
export PERSPECTIVE_API_KEY=...   # optional
python evaluation/evaluate.py --csv inference/Figstep/output_CSVs/text_unlearn_Figstep.csv --metrics detoxify perspective
```

Each inference script can also be run on its own, e.g.
`cd inference/PKU_RLHF && python run_inference.py --peft_model_id <adapter> --output_name my_run`.

## Data and models available upon request

We do **not** publish material that is harmful or explicit: the *Jailbreak in Pieces*
adversarial images, harmful responses generated by the original models (used as forget
targets for multi-modal unlearning), our generated outputs, and our trained adapters.
Researchers can get them by emailing **Trishna Chakraborty (tchak006@ucr.edu)** or
**Erfan Shayegani (sshay004@ucr.edu)**. See [data/README.md](data/README.md) for the full list.

## Citation

```bibtex
@inproceedings{chakraborty-etal-2024-textual,
    title = "Can Textual Unlearning Solve Cross-Modality Safety Alignment?",
    author = "Chakraborty, Trishna  and
      Shayegani, Erfan  and
      Cai, Zikui  and
      Abu-Ghazaleh, Nael B.  and
      Asif, M. Salman  and
      Dong, Yue  and
      Roy-Chowdhury, Amit  and
      Song, Chengyu",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.574/",
    doi = "10.18653/v1/2024.findings-emnlp.574",
    pages = "9830--9844"
}
```

## Acknowledgements

The unlearning code builds on [Large Language Model Unlearning](https://github.com/kevinyaobytedance/llm_unlearn)
(Yao et al., 2023, MIT License). We also use
[PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF),
[VQA-v2](https://visualqa.org/), [TruthfulQA](https://github.com/sylinrl/TruthfulQA),
[FigStep](https://github.com/ThuCCSLab/FigStep),
[JailBreakV-28K](https://github.com/EddyLuo1232/JailBreakV_28K) and
[Jailbreak in Pieces](https://openreview.net/forum?id=plmBsXHxgR).
