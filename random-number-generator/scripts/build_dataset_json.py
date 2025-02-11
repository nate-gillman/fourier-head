import json
import numpy as np
np.random.seed(42)
from pathlib import Path
import sys, os
import argparse

def generate_dataset(
        num_samples: int, 
        num_in_context_samples_per_prompt : int,
        distributions: list[tuple[float, float]], 
        output_path: str
    ):
    """
    Generate dataset of prompts and responses sampled from specified Gaussians
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    for idx in range(num_samples):

        # Cycle through distributions deterministically
        mu, sigma = distributions[idx % len(distributions)]

        # Get numerical samples
        samples = []
        for _ in range(num_in_context_samples_per_prompt+1):
            sample = -2
            while sample < -1.0 or sample > 1.0:
                sample = np.random.normal(mu, sigma)
            samples.append(sample)

        # Convert the numerical samples into a string, e.g. '-0.04, -0.24'
        in_context_samples_string = ", ".join(f"{sample:.2f}" for sample in samples[:-1])
        if len(samples[:-1]) > 0:
            in_context_samples_string += ", "
        
        data[str(idx)] = {
            "prompt": f"The following is a list of normally distributed random numbers in the interval [-1, 1] with mean {mu:.2f} and std {sigma:.2f}: {in_context_samples_string}",
            "response": f"{samples[-1]:.2f}",
            "distribution_info" : {
                "distribution_type": "gaussian",
                "mu": mu,
                "sigma": sigma,
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("dataset json saved to: ", output_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_in_context_samples_per_prompt', type=int)
    parser.add_argument('--train_set_size', type=int)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    # bounded to [-1, 1]
    distributions_train = [
        (-0.55, 0.10),
        (-0.03, 0.24),
        (0.42, 0.16),
        (0.55, 0.10),
    ]
    test_set_size=len(distributions_train)
    
    dataset_save_dir = f"data/{args.num_in_context_samples_per_prompt:02d}_in_context_samples"
    os.makedirs(dataset_save_dir, exist_ok=True)

    generate_dataset(args.train_set_size, args.num_in_context_samples_per_prompt, distributions_train, os.path.join(dataset_save_dir, "train.json"))
    generate_dataset(test_set_size, args.num_in_context_samples_per_prompt, distributions_train, os.path.join(dataset_save_dir, 'test_in_domain.json'))
    generate_dataset(test_set_size, args.num_in_context_samples_per_prompt, distributions_train, os.path.join(dataset_save_dir, 'val_in_domain.json'))

if __name__ == "__main__":
    main()