# Data

All paths below are the defaults used by [`paths.py`](../paths.py). Each one can be
overridden with the environment variable shown, so the datasets can live anywhere.

## Included in this repository

| File | Used by | Description |
|---|---|---|
| `vqa_val4000_llava1.5-7b_responses.csv` | multi-modal unlearning / SFT (normal data) | Responses of the original `llava-hf/llava-1.5-7b-hf` to the first 4000 questions of the VQA-v2 validation split (`Graphcore/vqa`). Env: `VQA_RESPONSES_CSV`. |

## Public data you need to download

| Data | Used by | How to get it | Default location / env var |
|---|---|---|---|
| **PKU-SafeRLHF** | textual unlearning (harmful text), PKU inference | Downloaded automatically from the HF Hub (`PKU-Alignment/PKU-SafeRLHF`). | – |
| **VQA-v2 (validation)** | multi-modal normal data, VQA inference | We used `Graphcore/vqa` from the HF Hub (~35 GB incl. COCO images), which is **no longer publicly available**. Use a VQA-v2 copy with the same schema (`question`, and `image_id` = path to the local COCO image) and the same validation order, and point to it with `VQA_HF_DATASET`. Our CSV below lists the 4000 questions in order so you can check alignment. | `VQA_HF_DATASET` |
| **TruthfulQA** | textual unlearning (normal text) | `wget -P data/ https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv` | `data/TruthfulQA.csv` · `TRUTHFULQA_CSV` |
| **FigStep (SafeBench, 500 images)** | FigStep inference, `--harmful_data figstep` | Clone [ThuCCSLab/FigStep](https://github.com/ThuCCSLab/FigStep) and copy `data/images/SafeBench/` | `data/FigStep/SafeBench/` · `FIGSTEP_IMAGES_DIR` |
| **JailBreakV-28K** | `--harmful_data jailbreakv` (images) | Follow [JailBreakV-28K](https://github.com/EddyLuo1232/JailBreakV_28K) / [HF dataset](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k). The folder must contain the sub-folders referenced in our CSV (e.g. `figstep/`, `llm_transfer_attack/`). | `data/JailBreakV_28K/` · `JAILBREAKV_ROOT` |

## Restricted data: available upon request

Some files contain **harmful content**: explicit or dangerous adversarial images, and
harmful text produced by the original (non-unlearned) models. We do not publish them.
Researchers can request them by email from
**Trishna Chakraborty (tchak006@ucr.edu)** or **Erfan Shayegani (sshay004@ucr.edu)**.
Please state your affiliation and intended research use.

Place them under `data/restricted/` (or set `VLMU_RESTRICTED_DIR`):

| File | Used by | Env var |
|---|---|---|
| `JailbreakV_llava1.5-7b_bfloat16_responses.csv`: harmful responses of LLaVA-1.5-7B to 2000 JailBreakV-28K prompts (first 500 used for training) | `--harmful_data jailbreakv` (unlearning and SFT) | `JAILBREAKV_RESPONSES_CSV` |
| `Figstep_all_llava1.5-7b_responses.csv`: harmful responses of LLaVA-1.5-7B to the 500 FigStep images | `--harmful_data figstep` (unlearning) | `FIGSTEP_RESPONSES_CSV` |
| `jailbreak_in_pieces_images/`: the 9 adversarial images of the *Jailbreak in Pieces* benchmark | `inference/JBPieces` | `JBPIECES_IMAGES_DIR` |
| Trained LoRA adapters (textual unlearning, multi-modal unlearning, SFT; LLaVA-1.5-7B and LLaVA-1.6-Mistral-7B) | `scripts/run_inference.sh` | – |
| Generated responses and logs for all experiments | – | – |

The textual-unlearning pipeline (`unlearn_harm_VLM.py`) and the PKU-SafeRLHF / VQA / FigStep
evaluations run with public data only.
