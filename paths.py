"""
Central place for all data locations used by the training and inference scripts.

Every path can be overridden with an environment variable, so you can keep the
(large) datasets anywhere on disk. See data/README.md for how to obtain each item.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VLMU_DATA_DIR", os.path.join(REPO_ROOT, "data"))

# --- Public data -----------------------------------------------------------------

# VQA-v2 (normal multi-modal data + VQA inference): the first 520 validation examples of
# lmms-lab/VQAv2, read from the first parquet shard only (~90 MB, images included).
# (The paper originally used `Graphcore/vqa`, which is no longer on the Hub; its validation
# order is identical to lmms-lab/VQAv2, so the saved responses below stay aligned.)
VQA_HF_DATASET = os.environ.get("VQA_HF_DATASET", "lmms-lab/VQAv2")
VQA_DATA_FILES = os.environ.get("VQA_DATA_FILES", "data/validation-00000-of-00068.parquet")
VQA_NUM_EXAMPLES = int(os.environ.get("VQA_NUM_EXAMPLES", 520))

# TruthfulQA (normal/benign text data for text-only unlearning).
TRUTHFULQA_CSV = os.environ.get("TRUTHFULQA_CSV", os.path.join(DATA_DIR, "TruthfulQA.csv"))

# Responses of the original LLaVA-1.5-7B to the first 520 VQA-v2 validation questions
# (row i <-> validation example i). Used as "normal" targets in multimodal unlearning / SFT.
VQA_RESPONSES_CSV = os.environ.get(
    "VQA_RESPONSES_CSV", os.path.join(DATA_DIR, "vqa_val520_llava1.5-7b_responses.csv")
)

# FigStep typographic images (SafeBench, 500 images) from the official FigStep repo.
FIGSTEP_IMAGES_DIR = os.environ.get(
    "FIGSTEP_IMAGES_DIR", os.path.join(DATA_DIR, "FigStep", "SafeBench")
)

# JailBreakV-28K images root (the folder that contains e.g. `figstep/`, `llm_transfer_attack/`).
JAILBREAKV_ROOT = os.environ.get("JAILBREAKV_ROOT", os.path.join(DATA_DIR, "JailBreakV_28K"))

# --- Restricted data (available upon request, see README) -------------------------

RESTRICTED_DIR = os.environ.get("VLMU_RESTRICTED_DIR", os.path.join(DATA_DIR, "restricted"))

# Harmful responses of the original LLaVA-1.5-7B to 2000 JailBreakV-28K prompts
# (columns: "JailbreakV Prompt", "JailbreakV Image Path", "JailbreakV Pretrained Response").
JAILBREAKV_RESPONSES_CSV = os.environ.get(
    "JAILBREAKV_RESPONSES_CSV",
    os.path.join(RESTRICTED_DIR, "JailbreakV_llava1.5-7b_bfloat16_responses.csv"),
)

# Harmful responses of the original LLaVA-1.5-7B to the 500 FigStep images
# (column: "Figstep Pretrained Response", rows aligned with the sorted FigStep file names).
FIGSTEP_RESPONSES_CSV = os.environ.get(
    "FIGSTEP_RESPONSES_CSV",
    os.path.join(RESTRICTED_DIR, "Figstep_all_llava1.5-7b_responses.csv"),
)

# Adversarial images used for the "Jailbreak in Pieces" evaluation.
JBPIECES_IMAGES_DIR = os.environ.get(
    "JBPIECES_IMAGES_DIR", os.path.join(RESTRICTED_DIR, "jailbreak_in_pieces_images")
)
