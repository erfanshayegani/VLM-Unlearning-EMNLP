
import argparse
import logging
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoProcessor,LlavaForConditionalGeneration, get_scheduler, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration # for llava/mistral
from PIL import Image

from model_utils import get_bnb_config, load_llava, load_processor, save_adapter


"""
I wrote multimodal_utils.py myself
"""
from multimodal_utils import (
    create_sft_dataloader, # creates the harmful and benign dataloader multimodal
    multimodal_get_answer_loss,
    create_sft_JB28k_dataloader
)


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    accelerator = Accelerator()
    device = accelerator.device

    # print(device)

    bnb_config = get_bnb_config()
    # llava-hf/llava-1.5-7b-hf or llava-hf/llava-v1.6-mistral-7b-hf (LlavaNext class is picked automatically)
    model = load_llava(args.model_name, quantization_config=bnb_config)

    # If use LoRA.
    # model.gradient_checkpointing_enable() # reduce number of stored activations - but it slows training by approximately 20%.
    """
    model = prepare_model_for_kbit_training(model)
    This prepares a PEFT model for training. 
    0- freezes base model's layers (makes Require_grad = False for all the params)
    1- Casts the layernorm in fp32
    2- makes output embedding layer require grads 
    3- casts input embeddings and output embeddings (lm head) to fp32
    """
    if args.use_lora:
        model = prepare_model_for_kbit_training(model) # important - I added this

        # for LLaVA - Only attach adapter to the LLM on top of LLaVA
        q_projs = [f'language_model.model.layers.{i}.self_attn.q_proj' for i in range(32)]
        v_projs = [f'language_model.model.layers.{i}.self_attn.v_proj' for i in range(32)]
        LLM_projs = q_projs + v_projs

        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            # target_modules=["q_proj", "v_proj"],
            target_modules= LLM_projs,
        )
        model = get_peft_model(model, peft_config)

    # model.to(device) # comment this if you use Quantization
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ### if the pad token exists! ################
    #### It's better to set padding = right for fine-tuning!
    # [sxxsxxsxPPPPP,
    #  saxsaxsacPPPPP
    #  scacsaccsacPPPP
    #  scacascsacascPPPPP]
    # tokenizer.pad_token = tokenizer.eos_token # for llama-2
    
    processor = load_processor(args.model_name)
    logging.info("model to be fine-tuned loaded successfully!")

    logging.info("Creating the dataloaders!")
    # Load harmful data.
    # good_dataloader, bad_dataloader = create_sft_dataloader(processor, batch_size = args.batch_size) # mutli-modal dataloader
    if args.harmful_data == "figstep":
        mixed_dataloader = create_sft_dataloader(processor, batch_size = args.batch_size) # FigStep + VQA (multi-modal)
    else:
        mixed_dataloader = create_sft_JB28k_dataloader(processor, batch_size = args.batch_size) # JailBreakV-28K + VQA (multi-modal)
    logging.info("The dataloaders loaded successfully!")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_sft_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        mixed_dataloader, # mixed - both bad and benign data in the batches!
        # good_dataloader, #
        # bad_dataloader, #
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, mixed_dataloader, lr_scheduler # note to replace the dataloader!
    )

    model.train()
    logging.info("model put in training mode!")
    # Reference model for computing KL.

    bad_loss = 0.0
    good_loss = 0.0
    idx = 0
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.

    # while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
    while idx < args.max_sft_steps:
        for mixed_batch in mixed_dataloader:
        # for good_batch, bad_batch in zip(good_dataloader, bad_dataloader):
            s = time.time()
            ############ GA on answer only. ############
            # good_loss = multimodal_get_answer_loss("gd", good_batch, model, processor, device=device)
            # bad_loss = multimodal_get_answer_loss("gd", bad_batch, model, processor, device=device)
            loss = multimodal_get_answer_loss("gd", mixed_batch, model, processor, device=device)

            # Final loss = bad loss + good loss.
            # loss = (
            #     args.good_weight * good_loss
            #     + args.bad_weight * bad_loss
            # )

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Print.
            stats = (
                f"batch: {idx}, "
                f"Fine-Tuning_loss: {loss:.2f}, "
                # f"BenignData_loss: {good_loss:.2f}, "
                # f"HarmfulData_loss: {bad_loss:.2f}, "
            )
            logging.info(stats)
            e = time.time()
            logging.info(f"Batch took Total time: {(e - s):.2f} Seconds") 
            logging.info("-----------------------------------")
            print(stats)
            idx += 1
            # Save model.
            # if idx % args.save_every == 0:
            #     model.save_pretrained(args.model_save_dir, from_pt=True)
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    # if args.use_lora:
    #     model = model.merge_and_unload()

    # Save final model.
    # model.save_pretrained(args.model_save_dir, from_pt=True)

    # accelerator does not support Push to Hub, so first, unwrap the model please!
    # only if you use accelerate launch sft.py:
    unwrapped_model = accelerator.unwrap_model(model)
    save_adapter(unwrapped_model, args.model_save_dir, args.push_to_hub)
    logging.info("SFT Finished!")
    logging.info(f"Adapter saved to {args.model_save_dir}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--max_sft_steps",
        type=int,
        # default=1000,
        default=4500,
        help="Max number of SFT steps.",
    )
    # parser.add_argument(
    #     "--bad_weight", type=float, default=1.0, help="Weight on the bad loss."
    # )

    # parser.add_argument(
    #     "--good_weight", type=float, default=1.0, help="Weight on the good loss."
    # )

    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of SFT."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="SFT LR.")
    # parser.add_argument(
    #     "--max_bad_loss",
    #     type=float,
    #     default=100,
    #     help="Maximum loss on bad samples to terminate.",
    # )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--harmful_data",
        type=str,
        choices=["jailbreakv", "figstep"],
        default="jailbreakv",
        help="Multi-modal harmful data mixed with VQA data for SFT: JailBreakV-28K or FigStep.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/llava1.5-7b_multimodal_sft",
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Optional Hugging Face Hub repo id to also push the adapter to.",
    )
    # parser.add_argument(
    #     "--model_save_dir",
    #     type=str,
    #     default="models/opt1.3b_SFT",
    #     help="Directory to save model.",
    # )
    # parser.add_argument(
    #     "--save_every", type=int, default=500, help="How many steps to save model."
    # )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
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
