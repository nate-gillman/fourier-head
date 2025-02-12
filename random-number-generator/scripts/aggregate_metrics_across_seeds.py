import os
import json
import numpy as np
from typing import Dict, List
import sys
from scipy.stats import norm

def compute_tvd(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Total Variation Distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p1 - p2))

def compute_metrics_for_distributions(
        true_dist: np.ndarray, pred_dist: np.ndarray, num_unique_samples: np.ndarray) -> Dict:
    """Compute TVD error for given distributions."""

    assert np.isclose(true_dist.sum(), 1.0) and np.isclose(true_dist.sum(), pred_dist.sum())
    
    tvd_error = compute_tvd(true_dist, pred_dist)

    return {
        'tvd_error': tvd_error,
        'predicted_distribution': pred_dist,
        'true_distribution': true_dist,
        'num_unique_samples': num_unique_samples
    }

def aggregate_metrics(seed_results: List[Dict]) -> Dict:
    """Aggregate metrics across seeds."""
    if not seed_results:
        return {}
    metrics = {}
    for test_idx in seed_results[0].keys():
        # Collect metrics for this test case across all seeds
        tvd_errors = []
        pred_distributions = []
        num_unique_samples = []
        for seed_result in seed_results:
            if test_idx in seed_result:
                result = seed_result[test_idx]
                tvd_errors.append(result['tvd_error'])
                pred_distributions.append(result['predicted_distribution'])
                num_unique_samples.append(result['num_unique_samples'])
        
        if not tvd_errors:
            continue
        
        # Calculate pairwise TVD variance
        pairwise_tvds = [
            compute_tvd(pred_distributions[i], pred_distributions[j])
            for i in range(len(pred_distributions))
            for j in range(i+1, len(pred_distributions))
        ]

        # Find distributions corresponding to min/median/max TVD
        sorted_idx = np.argsort(tvd_errors)
        min_idx = sorted_idx[0]
        median_idx = sorted_idx[len(sorted_idx)//2]
        max_idx = sorted_idx[-1]
        
        # Store aggregated metrics
        metrics[test_idx] = {
            'tvd_error_mean': float(np.mean(tvd_errors)),
            'tvd_error_sem': float(np.std(tvd_errors) / np.sqrt(len(tvd_errors))),
            'tvd_variance_mean': float(np.mean(pairwise_tvds) if pairwise_tvds else 0.0),
            'tvd_variance_sem': float(np.std(pairwise_tvds) / np.sqrt(len(pairwise_tvds))) if pairwise_tvds else 0.0,
            'num_unique_samples_mean': float(np.mean(num_unique_samples)),
            'num_unique_samples_sem': float(np.std(num_unique_samples)) / np.sqrt(len(num_unique_samples)),
            'n_seeds': len(seed_results),
            'true_distribution': seed_results[0][test_idx]['true_distribution'].tolist(),
            'predicted_distributions': {
                'min_tvd': {
                    'distribution': pred_distributions[min_idx].tolist(),
                    'tvd': float(tvd_errors[min_idx])
                },
                'median_tvd': {
                    'distribution': pred_distributions[median_idx].tolist(),
                    'tvd': float(tvd_errors[median_idx])
                },
                'max_tvd': {
                    'distribution': pred_distributions[max_idx].tolist(),
                    'tvd': float(tvd_errors[max_idx])
                }
            }
        }

    # Calculate aggregate statistics across all test cases
    tvd_means = [metrics[test_idx]['tvd_error_mean'] for test_idx in metrics]
    unique_samples_means = [metrics[test_idx]['num_unique_samples_mean'] for test_idx in metrics]
    
    # Add aggregate statistics
    metrics['agg'] = {
        'tvd_error_mean': float(np.mean(tvd_means)),
        'tvd_error_sem': float(np.std(tvd_means) / np.sqrt(len(tvd_means))),
        'num_unique_samples_mean': float(np.mean(unique_samples_means)),
        'num_unique_samples_sem': float(np.std(unique_samples_means) / np.sqrt(len(unique_samples_means)))
    }
    
    return metrics

def get_distributions_from_gaussian_data(test_data, n_bins=10):
    """Convert Gaussian parameters and samples into discretized distributions."""
    # Get parameters for true distribution
    true_mu = test_data['true_mu']
    true_sigma = test_data['true_sigma']
    
    # Get samples for empirical distribution
    samples = np.array(test_data['parsed_numbers'])
    valid_samples = samples[(samples >= -1) & (samples < 1)]
    
    # Create bin edges from 0 to 1
    bins = np.linspace(-1, 1, n_bins+1)
    
    # Create true distribution using CDF differences (bin integration)
    true_dist = np.array([
        norm.cdf(right, true_mu, true_sigma) - norm.cdf(left, true_mu, true_sigma)
        for left, right in zip(bins[:-1], bins[1:])
    ])
    true_dist = true_dist / np.sum(true_dist)  # Normalize
    
    # Create predicted distribution from histogram of valid samples
    pred_dist, _ = np.histogram(valid_samples, bins=bins, density=False) # array of counts; bins = bin edges in this fxn
    pred_dist = pred_dist / np.sum(pred_dist)  # pmf array

    num_unique_samples = test_data['num_unique_samples']
    
    return true_dist, pred_dist, num_unique_samples

def process_seed_directories(root_path: str) -> Dict[str, List[str]]:
    """Process seed directories and compute metrics."""
    # Collect directories by experiment
    experiment_dirs = {}
    for item in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, item)) and "-seed-" in item:
            exp_name = item.split("-seed-")[0]
            experiment_dirs.setdefault(exp_name, []).append(item)
    
    # Process each experiment
    for exp_name, dirs in experiment_dirs.items():
        seed_results = []
        
        for dir_name in dirs:
            json_path = os.path.join(root_path, dir_name, "inference_results_in_domain_metrics.json")

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Skipping {json_path} - file not found or invalid JSON")
                continue

            processed_results = {}
            for test_idx, test_data in data.items():
                if not isinstance(test_data, dict):
                    continue

                # Get distributions from Gaussian data
                true_dist, pred_dist, num_unique_samples = get_distributions_from_gaussian_data(test_data, n_bins=20)
                
                # Compute metrics
                processed_results[test_idx] = compute_metrics_for_distributions(
                    true_dist,
                    pred_dist,
                    num_unique_samples,
                )
            
            if processed_results:
                seed_results.append(processed_results)
        
        if seed_results:
            # Write aggregated metrics
            metrics = aggregate_metrics(seed_results)
            output_path = os.path.join(root_path, f"{exp_name}-aggregated.json")
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Wrote aggregated metrics to: {output_path}")
    
    return experiment_dirs

def main():
    exp_dir = sys.argv[1]
    process_seed_directories(exp_dir)

if __name__ == "__main__":
    main()