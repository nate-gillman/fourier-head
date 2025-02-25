from typing import Optional, Tuple
import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftModel
from peft.peft_model import PeftModelForCausalLM

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.utils.train_utils import train
from llama_recipes.configs import lora_config as LORA_CONFIG

# Go up two directories from the script location to reach the project root
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fourier_head import Fourier_Head
from torch.optim.lr_scheduler import LambdaLR
import os
import time

# adapted from https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/quickstart_peft_finetuning.ipynb

class Quantizer:
    """over the closed interval given, for k bins, assign a number in that bound to a bin.
    """
    def __init__(self, num_bins: int, lower_bound: float, upper_bound: float) -> None:
        self.num_bins = num_bins
        self.lb = lower_bound
        self.ub = upper_bound
        self._bin_edges = np.linspace(self.lb, self.ub, self.num_bins + 1)

    def get_bin_id_from_number(self, number: float) -> int:
        if not (self.lb <= number <= self.ub):
            msg = f"Number is out of bounds. Got {number}, must be in [{self.lb}, {self.ub}]"
            raise ValueError(msg)

        idx = np.searchsorted(self._bin_edges, number, side='right') - 1
        idx = np.clip(idx, 0, self.num_bins - 1)
        return idx
    
    def get_number_from_bin_id(self, bin_id: int) -> float:
        if not (0 <= bin_id < self.num_bins):
            raise ValueError(f"bin_id out of range. Got {bin_id}, must be in [0, {self.num_bins - 1}]")

        left_edge = self._bin_edges[bin_id]
        right_edge = self._bin_edges[bin_id + 1]

        # we can get more creative here, right now we return the midpoint of the bin
        # but could have other ways of mapping ID to number
        return 0.5 * (left_edge + right_edge)


def tokenize_phrases(prompt, response, tokenizer, quantizer: Optional[Quantizer] = None, max_length: int = 64):
    """
    Based off of https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/samsum_dataset.py
    """

    PADDING_TOKEN_ID = 0 # can be any valid token

    # STEP 1: build input_ids
    prompt_tokenized = tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False)
    if quantizer: # if we're using the Fourier head, then we don't want the EOS token...
        response_num = float(response)
        bin_id = quantizer.get_bin_id_from_number(response_num)
        response_tokenized = [bin_id]
    else:
        response_tokenized = tokenizer.encode(response +  tokenizer.eos_token, add_special_tokens=False)
    assert len(prompt_tokenized + response_tokenized) <= max_length
    padding_length = max_length - len(prompt_tokenized + response_tokenized)
    input_ids = prompt_tokenized + response_tokenized + padding_length * [PADDING_TOKEN_ID]

    # STEP 2: build attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * len(prompt_tokenized + response_tokenized) + [0] * padding_length

    # STEP 3: build labels (tokens labeled with -100 signal that the model shouldn't learn to predict those tokens)
    labels = [-100] * len(prompt_tokenized) + response_tokenized + [-100] * padding_length

    return dict(
        input_ids=torch.tensor(input_ids),
        attention_mask=torch.tensor(attention_mask),
        labels=torch.tensor(labels)
    )

def get_gaussian_dataloader(data_dir, tokenizer, split, quantize_vocabulary_size_to: Optional[int] = None, max_samples: Optional[int] = None, batch_size : Optional[int] = 1):
    
    # making max length a function of the dataset is more efficient than hard-coding a max length...
    max_length = 39 + 5*int(data_dir.split("/")[-1][:2]) + 5

    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test_in_domain.json")
    with open(train_path if split == 'train' else test_path) as f:
        dataset = json.load(f)
    
    quantizer = None
    if quantize_vocabulary_size_to is not None:
        # for now, hard-coded interval [-1, 1]
        quantizer = Quantizer(quantize_vocabulary_size_to, -1, 1)

    formatted_dataset = []
    for item in dataset.values():
        dataset_sample = tokenize_phrases(item['prompt'], item["response"], tokenizer, quantizer, max_length=max_length)
        formatted_dataset.append(dataset_sample)

    if max_samples is not None:
        # can use this just to set to a very small number during dry run tests
        formatted_dataset = formatted_dataset[:max_samples]
        warnings.warn(f"Loading the tuning dataset with a limited number of samples: {max_samples}. Please check if this is intended.")

    # convert to dataloader
    dataloader = torch.utils.data.DataLoader(formatted_dataset, num_workers=0, pin_memory=True, batch_size=batch_size)
    
    return dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_frequencies', type=int, help="Set to 0 for baseline linear head. Non-zero means Fourier Head and sets the number of frequencies.", default=0)
    parser.add_argument('--vocab_size', type=int, help="Set to the size of the vocabulary for the fourier head. Has no effect if num_frequencies is 0.", default=2048)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility", default=42)
    args = parser.parse_args()
    return args

def set_seed(seed: int) -> None:
    print("Setting seed to: ", seed)
    """Set random seed for reproducibility across PyTorch, NumPy, and Python's random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Optional: for complete reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FourierHeadLlama(LlamaForCausalLM):

    @classmethod
    def from_pretrained(
        cls,
        vocab_size: int,
        num_frequencies: int,
        *,
        pretrained_model_name_or_path,
        device_map,
        quantization_config,
        cache_dir,
        attn_implementation,
        torch_dtype,
    ):
        # do not call this if loading from disk a trained version of fourier
        base = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        base.config.vocab_size = vocab_size
        prev_lm_head = base.lm_head

        # --- swap ---
        base.lm_head = Fourier_Head(
            base.config.hidden_size, base.config.vocab_size, num_frequencies=num_frequencies, dtype=torch.float32
        )
        base.lm_head.to(base.device)

        del prev_lm_head
        return base

def save_fourier_head_llama_lora(lora_model: PeftModelForCausalLM, train_config, fourier_config) -> None:
    """Saves state and config required for loading a trained LLaMA + LoRA + Fourier Head.

    If run with experiment.sh, the folder would look something like:

    output_dir
    - adapter_config.json
    - adapter_model.safetensors
    - fourier_config.json
    - inference_results... .json
    - ...
    - inference_results ... .json
    - lm_head.bin
    - README.md
    - train_config.json

    """
    base_model = lora_model.base_model

    if not isinstance(base_model.lm_head, Fourier_Head):
        raise ValueError("This method may only be used to save PEFT models that use Fourier_Head as lm_head.")

    # save the adapters
    lora_model.save_pretrained(train_config.output_dir, save_embedding_layers=False)

    if not isinstance(base_model.lm_head, Fourier_Head):
        raise ValueError("LM Head is not an instance of Fourier Head.")
    
    # need custom logic for storing the weights of the fourier head
    fourier_head_state = base_model.lm_head.state_dict()
    torch.save(fourier_head_state, Path(train_config.output_dir) / "lm_head.bin")

    (Path(train_config.output_dir) / "train_config.json").write_text(
        json.dumps(asdict(train_config))
    )
    (Path(train_config.output_dir) / "fourier_config.json").write_text(
        json.dumps(fourier_config)
    )
    print(f"Training: saved PeftModel to path: {train_config.output_dir}")

def load_fourier_head_llama_lora(output_dir) -> Tuple[PeftModelForCausalLM, AutoTokenizer, Quantizer]:
    """Loads what has been saved to output dir by the function `save_fourier_head_llama_lora`.
    """
    train_conifg_string = (Path(output_dir) / "train_config.json").read_text()
    train_config = TRAIN_CONFIG(**json.loads(train_conifg_string))
    fourier_config = json.loads((Path(output_dir) / "fourier_config.json").read_text())

    config = BitsAndBytesConfig(load_in_8bit=True)
    
    # load base model
    model = FourierHeadLlama.from_pretrained(
        fourier_config["vocab_size"],
        fourier_config["num_frequencies"],
        # ...
        pretrained_model_name_or_path=train_config.model_name,
        device_map="auto",
        quantization_config=config,
        cache_dir="models",
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float32,
    )
    # load adapters
    peft_model = PeftModel.from_pretrained(model, output_dir)
    
    # load fourier weights
    model.lm_head.load_state_dict(torch.load(Path(output_dir) / "lm_head.bin"))
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quantizer = Quantizer(fourier_config["vocab_size"], -1, 1)

    return peft_model, tokenizer, quantizer

def output_dir_contains_fourier_model_p(output_dir) -> bool:
    tc = Path(output_dir) / "train_config.json"
    fc = Path(output_dir) / "fourier_config.json"
    lm = Path(output_dir) / "lm_head.bin"
    return all([x.exists() and x.is_file() for x in (tc, fc, lm)])

def train_LoRA(
        data_dir: str, 
        output_dir: str, 
        head_type: str = "linear",
        num_frequencies: int = 0, 
        vocab_size: Optional[int] = None, 
        num_epochs: int = 1, 
        seed: int = 42
    ) -> None:
    """
    Unified training function that handles both Fourier and linear head types.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save model outputs
        head_type: Either "linear" (default) or "fourier"
        num_frequencies: Number of frequencies for Fourier head (ignored if head_type="linear")
        vocab_size: Vocabulary size for Fourier head (ignored if head_type="linear")
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
    """
    if head_type not in ["linear", "fourier"]:
        raise ValueError(f"Invalid head_type. Got: {head_type}. Must be 'linear' or 'fourier'")
    if head_type == "fourier" and num_frequencies <= 0:
        raise ValueError(f"num_frequencies must be positive for Fourier head. Got: {num_frequencies}")

    set_seed(seed)

    train_config = TRAIN_CONFIG()
    train_config.output_dir = output_dir
    train_config.model_name = "meta-llama/Meta-Llama-3.1-8B-instruct"
    train_config.num_epochs = num_epochs
    train_config.batch_size_training = 64
    train_config.lr = 3e-4
    train_config.use_peft = True

    # Model initialization differs based on head type
    config = BitsAndBytesConfig(load_in_8bit=True)
    if head_type == "linear":
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            cache_dir="models",
            quantization_config=config,
            torch_dtype=torch.float32,
        )
    elif head_type == "fourier":  
        fourier_config = {
            'vocab_size': vocab_size,
            'num_frequencies': num_frequencies
        }
        model = FourierHeadLlama.from_pretrained(
            fourier_config['vocab_size'],
            fourier_config['num_frequencies'],
            # arguments for LlamaForCausalLM
            pretrained_model_name_or_path=train_config.model_name,
            device_map="auto",
            cache_dir="models",
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            quantization_config=config,
            torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration. 
    # target_modules = ["q_proj", "v_proj"] by default
    # ref: https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig
    lora_config = LORA_CONFIG()
    if head_type == "linear":
        peft_config = LoraConfig(**asdict(lora_config))
    elif head_type == "fourier":
        base_config = asdict(lora_config)
        fourier_lora_config = {
            **base_config
        }
        # do not track low rank updates on the fourier head, it should be trained from scratch 
        # with full params - no adapters
        fourier_lora_config["exclude_modules"] = "lm_head"
        peft_config = LoraConfig(**fourier_lora_config)

    model = prepare_model_for_kbit_training(model)
    lora_model: PeftModelForCausalLM = get_peft_model(model, peft_config)

    if head_type == "fourier":
        for param in lora_model.base_model.lm_head.parameters():
            param.requires_grad = True

    # dataloaders
    if head_type == "linear":
        train_dataloader = get_gaussian_dataloader(data_dir, tokenizer, 'train',
            batch_size=train_config.batch_size_training
        )
        eval_dataloader = get_gaussian_dataloader(data_dir, tokenizer, 'test',
            batch_size=train_config.batch_size_training
        )
    elif head_type == "fourier":
        train_dataloader = get_gaussian_dataloader(data_dir, tokenizer, 'train', 
            quantize_vocabulary_size_to=fourier_config['vocab_size'], 
            batch_size=train_config.batch_size_training
        )
        eval_dataloader = get_gaussian_dataloader(data_dir, tokenizer, 'test', 
            quantize_vocabulary_size_to=fourier_config['vocab_size']
        )

    optimizer = optim.AdamW(lora_model.parameters(), lr=train_config.lr)

    # scheduler; decays from train_config.lr at epoch=1 to train_config.lr/10 at epoch=train_config.num_epochs
    lambda_fn = lambda epoch: 1.0 - (0.9 * epoch/train_config.num_epochs)
    scheduler = LambdaLR(optimizer, lambda_fn)

    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    results = train(
        lora_model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None, None, None,
        wandb_run=None,
    )

    # save the model
    if head_type == "linear":
        lora_model.save_pretrained(train_config.output_dir)
    elif head_type == "fourier":
        save_fourier_head_llama_lora(lora_model, train_config, fourier_config)

        # try loading the model from disk after saving
        loaded_peft, _, quantizer = load_fourier_head_llama_lora(train_config.output_dir)
        base_model_loaded_from_disk = loaded_peft.base_model
        assert isinstance(base_model_loaded_from_disk.lm_head, Fourier_Head)
        assert base_model_loaded_from_disk.lm_head.dim_output == quantizer.num_bins

        # check that the parameters are more or less the same after saving and loading
        # if this is true, then our custom fourier head network can be saved and loaded from disk
        for p1, p2 in zip(lora_model.base_model.lm_head.parameters(), base_model_loaded_from_disk.lm_head.parameters()):
            # down cast to same quantization level for comparison
            if not torch.allclose(p1.to(torch.float32), p2.to(torch.float32), rtol=1e-5, atol=1e-8):
                raise AssertionError("Module parameters are not equal.")
        
    # if we reach here, all seems ok. 
    print("train_lora_model successfully completed!")

def main() -> None:
    
    args = parse_args()

    head_type = "fourier" if args.num_frequencies > 0 else "linear"

    start_time = time.time()

    train_LoRA(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        head_type=head_type,
        num_frequencies=args.num_frequencies,
        vocab_size=args.vocab_size,
        num_epochs=args.num_epochs,
        seed=args.seed
    )

    print(f"\nExecution time for train_LoRA: {(time.time() - start_time)/60} minutes\n")

if __name__ == "__main__":
    main()