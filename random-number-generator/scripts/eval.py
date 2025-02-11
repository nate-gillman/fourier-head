from typing import Optional
import argparse
import os
import re
import json

import numpy as np
from scipy.stats import norm

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    KL divergence between two Gaussians; (mu_1, sigma_1) is true, (mu_2, sigma_2) is model
    """
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5

def extract_first_number(sample: str) -> Optional[float]:
    """Extract number after '2.' in sample"""

    if not sample:
        return None
        
    delimiters = [", ", " "]

    if sample[0] == " ":
        sample = sample[1:]
    
    for delimiter in delimiters:
        try:
            first_part = sample.split(delimiter)[0]
            return float(first_part)
        except (ValueError, IndexError):
            continue
            
    print(f"Failed to extract number from sample: {sample!r}")
    return None

def evaluate_samples(results_file):
    with open(results_file) as f:
        data = json.load(f)
    
    stats = {}
    kl_divs = []
    for idx, item in data.items():

        if not idx.isdigit():
            continue

        assert item["distribution_info"]["distribution_type"] == "gaussian"
        true_mu = item["distribution_info"]['mu']
        true_sigma = item["distribution_info"]['sigma']
        
        # Extract numbers from samples
        numbers = []
        for sample in item['samples']:
            num = extract_first_number(sample)
            if num is not None:
                numbers.append(num)

        if numbers:

            # METRIC 1: mean, std
            sample_mu = np.mean(numbers)
            sample_sigma = np.std(numbers, ddof=1)

            # METRIC 2: KL divergence
            kl_div = kl_divergence_gaussian(true_mu, true_sigma, sample_mu, sample_sigma)
            kl_divs.append(kl_div)
            
            # METRIC 3: number of unique samples
            # observation: models with low KLD might repeat the same value a few times, even with as low as 20 samples
            num_unique_samples = len(set(numbers))

            stats[idx] = {
                'true_mu': true_mu,
                'true_sigma': true_sigma,
                'sample_mu': sample_mu,
                'sample_sigma': sample_sigma,
                'kl_divergence': kl_div,
                'num_valid_samples': len(numbers),
                'num_unique_samples': num_unique_samples,
                'parsed_numbers': numbers,
            }

    avg_kl = np.mean(kl_divs) if kl_divs else float('inf')
    stats["avg_kl"] = avg_kl

    # write metrics to new file
    results_file = results_file.split(".json")[0] + "_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print("Wrote results to: ", results_file)
    
    return stats, avg_kl

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--test_split', type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    test_split = args.test_split

    input_fname = os.path.join(output_dir, f"inference_results_{test_split}.json")
    stats, avg_kl  = evaluate_samples(input_fname)

    print("avg_kl = ", avg_kl)