"""
Helpers for loading the LLaVA models used in the paper.

- llava-hf/llava-1.5-7b-hf          -> LlavaForConditionalGeneration + AutoProcessor
- llava-hf/llava-v1.6-mistral-7b-hf -> LlavaNextForConditionalGeneration + LlavaNextProcessor
"""
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)


def is_llava_next(model_name: str) -> bool:
    name = model_name.lower()
    return "v1.6" in name or "next" in name


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization used for all experiments."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_llava(model_name: str, quantization_config=None):
    cls = LlavaNextForConditionalGeneration if is_llava_next(model_name) else LlavaForConditionalGeneration
    return cls.from_pretrained(model_name, quantization_config=quantization_config)


def load_processor(model_name: str):
    if is_llava_next(model_name):
        return LlavaNextProcessor.from_pretrained(model_name)
    return AutoProcessor.from_pretrained(model_name)


def save_adapter(model, save_dir: str, push_to_hub: str = None):
    """Save the (LoRA) adapter locally and optionally push it to the Hugging Face Hub."""
    if save_dir:
        model.save_pretrained(save_dir)
    if push_to_hub:
        model.push_to_hub(push_to_hub)
