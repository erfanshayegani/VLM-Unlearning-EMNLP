# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
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
    create_harmful_dataloader, # creates the harmful dataloader multimodal
    multimodal_get_answer_loss,
    create_normal_dataloader,
    get_vqa_answers_plaintext,
    multimodal_compute_kl,
    mutlimodal_get_rand_ans_loss,
    create_harmful_JB28k_dataloader
)


from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
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
    # tokenizer.pad_token = tokenizer.eos_token # for llama-2
    
    processor = load_processor(args.model_name)
    logging.info("model to be unlearned loaded successfully!")

    logging.info("Creating the dataloaders!")
    # Load harmful data.

    if args.harmful_data == "figstep":
        train_bad_loader = create_harmful_dataloader(processor, batch_size = args.batch_size) # FigStep (multi-modal)
    else:
        train_bad_loader = create_harmful_JB28k_dataloader(processor, batch_size = args.batch_size) # JailBreakV-28K (multi-modal)

    # train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    # train_bad_loader = create_pku_dataloader_from_dataset(
    #     processor.tokenizer, train_dataset, batch_size=args.batch_size
    # )
    
    # Get normal data.
    train_normal_loader = create_normal_dataloader(processor, batch_size = args.batch_size)
    # train_normal_loader, _, _ = create_truthfulqa_dataloader(
    #     processor.tokenizer, batch_size=args.batch_size
    # )

    logging.info("Both bad and normal dataloaders loaded successfully!")

    # erfan
    # Load normal answer used for random mismatch.
    # normal_ans = get_truthfulQA_answers_plaintext() 
    # normal_ans = ["I cannot assist with this!",
    #               "If you're experiencing thoughts of harming yourself or others, please see a mental health professional.",
    #               "I'm sorry, but I can't engage in or provide advice on activities that promote harm or violence.",
    #               "Let's direct the conversation to more peaceful content.",
    #               "instead of harm, spread love!!",
    #               "Enough with harm, violence, and drugs. Focus on positive things!",
    #               "This content is not good.",
    #               "These are bad things. Let's use them in a psotive way and help others!"] # erfan map to more helpful responses, not just random!
    # normal_ans = ["I cannot assist with this! END."]
    # normal_ans = ["I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>",
    #               "I cannot assist with this!<pad><pad><pad><pad><pad><pad><pad><pad></s>"]

    normal_ans = ["I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
                  "I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>"]


    
    # normal_ans = ["I cannot assist with this!</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "If you're experiencing thoughts of harming yourself or others, please see a mental health professional.</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "I'm sorry, but I can't engage in or provide advice on activities that promote harm or violence.</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "Let's direct the conversation to more peaceful content.</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "instead of harm, spread love!!</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "Enough with harm, violence, and drugs. Focus on positive things!</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "This content is not good.</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "These are bad things. Let's use them in a psotive way and help others!</s></s></s></s></s></s></s></s></s></s></s></s>",
    #               "Do not focus on harmful things. Focus on good</s></s></s></s></s></s></s></s></s></s></s></s>."]

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    model.train()
    logging.info("model put in training mode!")
    # Reference model for computing KL.
    pretrained_model = load_llava(args.model_name, quantization_config=bnb_config)
    logging.info("pretrained_model loaded successfully!")

    # pretrained_model.to(device) # comment this if you use Quantization

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.

    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            s = time.time()
            ############ GA on answer only. ############
            bad_loss = multimodal_get_answer_loss("ga", bad_batch, model, processor, device=device)

            ############ Random mismatch. ############
            # random_loss = mutlimodal_get_rand_ans_loss(
            #     bad_batch,
            #     normal_ans,
            #     model,
            #     processor,
            #     K=2,
            #     device=device,
            # )

            ############ KL on normal samples. ############
            normal_loss = multimodal_compute_kl(pretrained_model, model, normal_batch, device)

            # Final loss = bad loss + random smoothing + normal loss.
            loss = (
                args.bad_weight * bad_loss
                # + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                # f"GoodAnswer_loss: {random_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
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
    # only if you use accelerate launch unlearn.py:
    unwrapped_model = accelerator.unwrap_model(model)
    save_adapter(unwrapped_model, args.model_save_dir, args.push_to_hub)
    logging.info("Unlearning Finished!")
    logging.info(f"Adapter saved to {args.model_save_dir}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
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
        help="Multi-modal harmful data to unlearn: JailBreakV-28K (500 samples) or FigStep (500 images).",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/llava1.5-7b_multimodal_unlearned",
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Optional Hugging Face Hub repo id to also push the adapter to.",
    )
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
