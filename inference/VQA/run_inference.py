import argparse
import logging
import random
import time
import numpy as np
from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor,LlavaForConditionalGeneration, get_scheduler, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration # for llava/mistral
from PIL import Image
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root
from model_utils import get_bnb_config, load_llava, load_processor  # noqa: E402
from multimodal_utils import load_vqa_subset  # noqa: E402

from Inference_Utils import batch_generate_response  # noqa: E402  (batch generation helper)

def main(args) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    # print(device.index)

    bnb_config = get_bnb_config()

    # Original (pretrained) model, used as the reference in every output CSV.
    pretrained_model = load_llava(args.base_model, quantization_config=bnb_config)
    processor = load_processor(args.base_model)
    logging.info("Pretrained Model loaded successfully")

    # Unlearned / SFT model = base model + LoRA adapter (local directory or HF Hub id).
    model = load_llava(args.base_model, quantization_config=bnb_config)
    model = PeftModel.from_pretrained(model, args.peft_model_id)

    logging.info("Unlearned Model loaded successfully") 

    ########################
    logging.info("VQA Validation Set Benchmark")   

    # first 520 VQA-v2 validation examples (lmms-lab/VQAv2, first parquet shard)
    vqa_dataset_prompts, vqa_images = load_vqa_subset()

    start_time = time.time()

    unlearned = []
    pretrained = []
    total_prompts = []
    instruction_prompt = "Answer the question using a single word or phrase. "

    Num = 5 # Batch Size
    for i in range(0, (len(vqa_dataset_prompts) + Num - 1) // Num): # Number of Batches
        logging.info(f"{i}th batch started!")
        s = time.time() 
        batch_prompts = [f"### Question:<image>\n {instruction_prompt}{question}\n ### Answer:" for question in vqa_dataset_prompts[i * Num : (i+1) * (Num)]]
        total_prompts += batch_prompts

        unlearned += batch_generate_response(batch_prompts, model, processor, device, new_tokens=100, images=vqa_images[i * Num : (i+1) * (Num)])
        torch.cuda.empty_cache() # probably helpful - to get the unlearned stuff out of GPU and then do for the pretrained!
        pretrained += batch_generate_response(batch_prompts, pretrained_model, processor, device, new_tokens=100, images=vqa_images[i * Num : (i+1) * (Num)])
        torch.cuda.empty_cache()
        e = time.time()
        logging.info(
            (f"{i}th batch completed!", 
             f"It took Total time: {(e - s):.2f} Seconds")
             ) 
        logging.info(f"Batch Size: {Num}")
        logging.info("-----------------------------------")
      

    df = pd.DataFrame(list())
    df["VQA Prompt"] = total_prompts
    df["VQA Unlearned Response"] = unlearned
    df["VQA Pretrained Response"] = pretrained
    df = df.replace('', "I can't help!")  # we need to replace empty strings with sth else, otherwise they become NaN when we do to_csv
    settings = args.output_name

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, settings + ".csv"), index=False)
    logging.info("Generated Responses - Done!")     

    # stats = (
    #     f"batch: {idx}, "
    #     f"bad_loss: {-bad_loss:.2f}, "
    #     f"current_div_loss: {normal_loss:.2f}, "
    # )
    # logging.info(stats)

    end_time = time.time()
    logging.info(f"Total time: {((end_time - start_time) / 60):.2f} Mins")
    logging.info("Finished")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--peft_model_id",
        type=str,
        required=True,
        help="LoRA adapter to evaluate: a local directory (e.g. ../../models/llava1.5-7b_multimodal_unlearned) or a HF Hub id.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Base model the adapter was trained on (llava-hf/llava-1.5-7b-hf or llava-hf/llava-v1.6-mistral-7b-hf).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Prefix of the CSV file(s) written to --output_dir.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output_CSVs", help="Where to write the response CSVs."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="inference.log",
        help="Log file name",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
