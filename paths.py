"""
Central place for all data locations used by the training and inference scripts.

Every path can be overridden with an environment variable, so you can keep the
(large) datasets anywhere on disk. See data/README.md for how to obtain each item.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("VLMU_DATA_DIR", os.path.join(REPO_ROOT, "data"))

# --- Public data -----------------------------------------------------------------

# Hugging Face dataset used for VQA-v2 (normal multi-modal data + VQA inference).
# NOTE: `Graphcore/vqa` (used in the paper) is no longer publicly available on the Hub.
# The code expects its schema: columns `question` and `image_id`, where `image_id` is a
# path to a local COCO image file. See data/README.md.
VQA_HF_DATASET = os.environ.get("VQA_HF_DATASET", "Graphcore/vqa")

# TruthfulQA (normal/benign text data for text-only unlearning).
TRUTHFULQA_CSV = os.environ.get("TRUTHFULQA_CSV", os.path.join(DATA_DIR, "TruthfulQA.csv"))

# Responses of the original LLaVA-1.5-7B on the first 4000 VQA-v2 validation questions
# (Graphcore/vqa). Used as "normal" targets for the KL term in multimodal unlearning / SFT.
VQA_RESPONSES_CSV = os.environ.get(
    "VQA_RESPONSES_CSV", os.path.join(DATA_DIR, "vqa_val4000_llava1.5-7b_responses.csv")
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
