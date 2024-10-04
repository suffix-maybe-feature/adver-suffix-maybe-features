import time
import json
import os
import torch
import sys
import argparse
import logging
import numpy as np
import random
import shutil
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to sys.path
os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from latent_at import *

def setup_logger(log_directory, log_filename):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(log_directory, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def main(args):
    # Setup
    set_seed(args.seed)
    load_dotenv()
    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
    
    # Logger setup
    logger = setup_logger(args.log_dir, args.log_file)
    
    # Model setup
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_access_token,
        torch_dtype=torch.bfloat16
    ).to(args.device)
    
    if "Llama-2" in args.model_name or 'base' in args.model_name:
        model_type = "llama2"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Llama-3" in args.model_name:
        model_type = "llama3"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "zephyr" in args.model_name or "mistral" in args.model_name:
        model_type = "zephyr"    
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    else:
        raise Exception("Unsupported model type")
    

    
    # Prompt templates
    if model_type == "llama2":
        use_tokenizer_template = False
        custom_prompt_template = "[INST] {prompt} [/INST] \n"
        custom_completion_template = "{completion}"
    elif model_type == "llama3":
        use_tokenizer_template = False
        custom_prompt_template = "<|start_header_id|>user<|end_header_id|\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        custom_completion_template="{completion}"
    else:
        use_tokenizer_template = False
        custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
        custom_completion_template="{completion}"
    
    # Dataset and dataloader setup
    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="harmful-dataset",
        adv_column="rejected",
        def_column="chosen",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt="",
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

    dataloader = iter(lat_dataloader)
    
    # Create or clear output folder
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)
    logger.info(f'Folder {args.output_folder} has been created.')
    
    # Load embedding space
    if args.embedding_space_path is not None:
        embedding_space = np.load(args.embedding_space_path)
        logger.info(f"Embedding shape: {embedding_space.shape}")

    else:
        embedding_space = np.random.rand(32000, 4096)
    
    # Prepare for attack
    model.eval()
    freeze_parameters(model)
    
    data = []
    
    # Main loop
    jailbreak_count = 0
    logger.info(f"Current parameters: model_loss_parameter={args.model_loss_parameter}, learning_rate={args.learning_rate}, loss_parameter={args.loss_parameter}")
    count = 0
    for batch in dataloader:
        losses, wrappers, jailbroken = embedding_suffix_attack(
            logger=logger,
            batches=batch,
            model=model,
            model_layers_module="model.layers",
            layer=["embedding"],
            epsilon=args.epsilon,
            l2_regularization=args.l2_regularization,
            learning_rate=args.learning_rate,
            pgd_iterations=args.pgd_iterations,
            loss_coefs={"toward": args.loss_parameter, "away": 0.5},
            log_loss=True,
            return_loss_over_time=True,
            device=args.device,
            embedding_constraint=False,
            embedding_method='model',
            embedding_space=embedding_space,
            tokenizer=tokenizer,
            model_parameter=args.model_loss_parameter
        )
        if jailbroken:
            jailbreak_count += 1
        logger.info("***ADVERSARIAL LOSSES OVER TIME***")
        logger.info([round(l['adv_total'], 4) for l in losses])
        for wrapper in wrappers:
            wrapper.enabled = True
        with torch.no_grad():
            outputs = model.generate(
                batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]].unsqueeze(0).to(args.device),
                max_length=batch["adv_padd_prompt_tokens"].shape[1] + 200,
                do_sample=False
            )

        logger.info("***ATTACKED PERFORMANCE***")
        prompt = tokenizer.decode(batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]]).replace('\n', '')
        logger.info(f"Prompt: {prompt}")
        prompt_response = tokenizer.decode(outputs[0]).replace('\n', '')
        logger.info(f"Completion: {prompt_response[len(prompt):]}")
        data.append({
            'losses': [round(l['adv_total'], 4) for l in losses],
            'prompt': prompt,
            'prompt_response': prompt_response[len(prompt):],
            'params': {
                'model_loss_parameter': args.model_loss_parameter,
                'learning_rate': args.learning_rate,
                'loss_parameter': args.loss_parameter
            }
        })
        with open(os.path.join(args.output_folder, 'output.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        count += 1
        if count >= args.max_batches:
            break
    logger.info(f"Final jailbreak count: {jailbreak_count}")
    logger.info("*" * 100)

    # Save the parameters to a file
    params = {
        'model_loss_parameter': args.model_loss_parameter,
        'learning_rate': args.learning_rate,
        'loss_parameter': args.loss_parameter,
        'jailbreak_count': jailbreak_count
    }
    with open(os.path.join(args.output_folder, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Space Adversarial Attack")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default='/root/autodl-tmp/at_final/logs', help="Log directory")
    parser.add_argument("--log_file", type=str, default='experiment.log', help="Log file name")
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/llama2/base/model", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--output_folder", type=str, default='/root/autodl-tmp/at_final/uap', help="Output folder path")
    parser.add_argument("--embedding_space_path", type=str, default=None, help="Path to embedding space file")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for dataloader")
    parser.add_argument("--epsilon", type=float, default=0.5496, help="Attack L2 constraint")
    parser.add_argument("--l2_regularization", type=float, default=0.0, help="L2 regularization coefficient")
    parser.add_argument("--pgd_iterations", type=int, default=1000, help="Number of PGD iterations")
    parser.add_argument("--max_batches", type=int, default=10, help="Maximum number of batches to process")
    parser.add_argument("--model_loss_parameter", type=int, default=200, help="Model loss parameter")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--loss_parameter", type=float, default=1.5, help="Loss parameter")
    
    args = parser.parse_args()
    
    main(args)