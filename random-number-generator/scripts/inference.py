from typing import Optional
import os
import argparse
import json
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
from peft import PeftModel
from peft.peft_model import PeftModelForCausalLM

from train_LoRA import output_dir_contains_fourier_model_p, load_fourier_head_llama_lora, Quantizer

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'data/Times_New_Roman.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams.update({
    'font.size': 11,          # Adjust global font size
    'axes.titlesize': 14,     # Adjust title size
    'axes.labelsize': 12,     # Adjust axis label size
    'legend.fontsize': 10,    # Adjust legend font size
    'xtick.labelsize': 10,    # X-axis tick label size
    'ytick.labelsize': 10,    # Y-axis tick label size
    'axes.linewidth': 1.0,    # Adjust axes line width
    'grid.alpha': 0.6,        # Gridline transparency
})

def load_model_and_tokenizer(train_config, LoRA_model=None):
    config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        device_map="auto",
        quantization_config=config,
        cache_dir="models",
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float16,
    )
    
    if LoRA_model[0]:
        # Load PEFT adapter
        assert Path(LoRA_model[1]).exists()
        print(f"Inference: Loading PeftModel from path: {LoRA_model[1]}")
        model = PeftModel.from_pretrained(model, LoRA_model[1])
    
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def repeat_model_input_into_a_batch(model_input : dict, batch_size : int):

    model_input_batch = {
        key: tensor.repeat(batch_size, *([1] * (len(tensor.shape) - 1)))
        for key, tensor in model_input.items()
    }

    return model_input_batch

def visualize_pmfs(bin_centerpoints, categorical_distributions, save_learned_pmfs_info):

    # STEP 1: get values to graph for categorical distribution
    data_learned_pmf = categorical_distributions[0][0]
    riemann_sum_learned_pmf = (data_learned_pmf.sum() * (bin_centerpoints[1] - bin_centerpoints[0])).item()

    # STEP 2: get values to graph for ground truth PDF
    distribution_type = save_learned_pmfs_info["distribution_info"]["distribution_type"]
    if distribution_type == "gaussian":
        mu      = save_learned_pmfs_info["distribution_info"]["mu"]
        sigma   = save_learned_pmfs_info["distribution_info"]["sigma"]
        x_vals  = np.linspace(-1, 1, 2001)
        data_gt_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
        riemann_sum_gt_pdf = data_gt_pdf.sum() * (x_vals[1] - x_vals[0])
        data_gt_pdf = data_gt_pdf * riemann_sum_learned_pmf / riemann_sum_gt_pdf

        # normalize so they can be graphed together
        # data_gt_pdf = data_gt_pdf * data_learned_pmf.cpu().numpy().mean() / data_gt_pdf.mean()
        png_name = f"gaussian_mu_{mu}_sigma_{sigma}.png"
    else:
        raise NotImplementedError

    # STEP 3: graph them
    plt.figure(figsize=(10, 6))

    # Plot learned Fourier PMF as bars
    plt.bar(bin_centerpoints.cpu(), data_learned_pmf.cpu(), width=(bin_centerpoints[1]-bin_centerpoints[0]).item(), color='tab:blue', alpha=0.7, label='Learned Fourier PMF')

    # Plot ground truth PDF as line
    plt.plot(x_vals, data_gt_pdf, color='tab:green', linewidth=2, label='Ground Truth PDF')

    title = "PROMPT:\n" + save_learned_pmfs_info["prompt"]
    plt.title(title)
    plt.xlabel('Latent value')
    plt.ylabel('Probability Density')
    plt.grid(True, linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    os.makedirs(save_learned_pmfs_info["output_dir"], exist_ok=True)
    fname = os.path.join(save_learned_pmfs_info["output_dir"], png_name)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

    return None

def run_batch_inference(model: PeftModelForCausalLM, tokenizer, input_file, output_file, num_samples=5, quantizer: Optional[Quantizer]=None):
    with open(input_file, 'r') as f:
        prompts_data = json.load(f)
    
    results = {}
    model.eval()
    
    for idx, item in tqdm(prompts_data.items(), desc="Processing prompts"):
        samples = []
        prompt = item['prompt']
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
        BATCH_SIZE_INFERENCE = 32
        model_input_batch = repeat_model_input_into_a_batch(model_input, BATCH_SIZE_INFERENCE)
        
        for i in range(int(np.ceil(num_samples/BATCH_SIZE_INFERENCE))):
            with torch.inference_mode():

                if quantizer: # unbiased sampling if there's a quantizer

                    if i == 0:
                        # pass forward information needed to save and visualize
                        save_learned_pmfs = {
                            "output_dir"            : "/".join(output_file.split("/")[:-1]) + "/learned_pmfs_" + input_file.split("/")[-1].split(".")[0],
                            "save_pmfs_function"    : visualize_pmfs,
                            "distribution_info"     : item['distribution_info'],
                            "prompt"                : item["prompt"],
                        }
                        model.base_model.model.lm_head.save_learned_pmfs = save_learned_pmfs

                    output = model.generate( # this one we run last so that it writes the correct things to disk...
                        **model_input_batch, # model_input_batch["input_ids"].shape = (32, 37)
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        top_k=quantizer.num_bins, # vocab_size
                    ) # (32, 38)
                    model.base_model.model.lm_head.save_learned_pmfs = None

                    # because we set max_new_tokens=1, the last token returned corresponds to the sample.
                    # notice how output.shape[1] is equal to context.shape[1] + 1
                    fourier_vocab_token_ids = output[:, -1]
                    for fourier_vocab_token_id in fourier_vocab_token_ids:
                        fourier_number : float = quantizer.get_number_from_bin_id(fourier_vocab_token_id)
                        response = str(fourier_number) # convert back to text for consistency
                        samples.append(response)

                else:
                    
                    output = model.generate(
                        **model_input_batch,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                    )

                    first_new_idx = model_input_batch["input_ids"].shape[1]
                    for batch_elt in range(output.shape[0]):
                        response = tokenizer.decode(output[batch_elt, first_new_idx:], skip_special_tokens=True)
                        samples.append(response)
        
        results[idx] = {
            **item,
            "samples": samples
        }
    
        # write intermediate results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--test_split', type=str)
    parser.add_argument('--is_LoRA_model', type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    is_LoRA_model = args.is_LoRA_model
    data_dir = args.data_dir
    output_dir = args.output_dir
    test_split = args.test_split

    output_fname = os.path.join(output_dir, f"inference_results_{test_split}.json")

    quantizer = None
    if output_dir_contains_fourier_model_p(output_dir):
        # fourier models
        model, tokenizer, quantizer = load_fourier_head_llama_lora(output_dir)
    else:
        # non-fourier models
        train_config = TRAIN_CONFIG()
        train_config.model_name = "meta-llama/Meta-Llama-3.1-8B-instruct" # the base model
        train_config.use_fast_kernels = True
        train_config.use_fp16 = True
        train_config.use_peft = True
        model, tokenizer = load_model_and_tokenizer(
            train_config, 
            LoRA_model=(is_LoRA_model, output_dir)
        )
    
    test_split_fname =  os.path.join(data_dir, f"test_{test_split}.json")
    run_batch_inference(
        model,
        tokenizer,
        input_file=test_split_fname,
        output_file=output_fname,
        num_samples=128,
        quantizer=quantizer,
    )

if __name__ == "__main__":
    """Run with:

        python inference.py --output_dir output/test --test_split in_domain --is_LoRA_model True
    """
    main()