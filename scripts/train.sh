#!/usr/bin/env bash
# Training commands for the experiments in the paper.
# Run from the repository root:  bash scripts/train.sh
# Each command saves a LoRA adapter to --model_save_dir. Comment out what you don't need.
set -e
mkdir -p logs models

LLAVA15=llava-hf/llava-1.5-7b-hf
LLAVA16=llava-hf/llava-v1.6-mistral-7b-hf

########## 1) Textual unlearning (text-only harmful data: PKU-SafeRLHF; normal data: TruthfulQA) ##########
python unlearn_harm_VLM.py --use_lora --model_name $LLAVA15 \
    --bad_weight 0.05 --random_weight 1 --normal_weight 1 \
    --model_save_dir models/llava1.5-7b_text_unlearned \
    --log_file logs/text_unlearn_llava1.5-7b.log

python unlearn_harm_VLM.py --use_lora --model_name $LLAVA16 \
    --bad_weight 0.05 --random_weight 1 --normal_weight 1 \
    --model_save_dir models/llava1.6-mistral-7b_text_unlearned \
    --log_file logs/text_unlearn_llava1.6-mistral-7b.log

########## 2) Multi-modal unlearning (image+text harmful data; normal data: VQA-v2) ##########
python multimodal-unlearn_harm_VLM.py --use_lora --model_name $LLAVA15 --harmful_data jailbreakv \
    --bad_weight 0.05 --normal_weight 1 \
    --model_save_dir models/llava1.5-7b_multimodal_unlearned \
    --log_file logs/multimodal_unlearn_llava1.5-7b.log

########## 3) Multi-modal SFT baseline (harmful data -> refusal, mixed with VQA-v2) ##########
python multimodal-SFT_harm_VLM.py --use_lora --model_name $LLAVA15 --harmful_data jailbreakv \
    --model_save_dir models/llava1.5-7b_multimodal_sft \
    --log_file logs/multimodal_sft_llava1.5-7b.log

########## (optional) Text-only LLM unlearning, e.g. Llama-2-7B ##########
# accelerate launch unlearn_harm.py --use_lora --model_name meta-llama/Llama-2-7b-hf \
#     --bad_weight 0.5 --random_weight 1 --normal_weight 1 \
#     --model_save_dir models/llama2-7b_unlearned --log_file logs/llama2-7b_unlearn.log
