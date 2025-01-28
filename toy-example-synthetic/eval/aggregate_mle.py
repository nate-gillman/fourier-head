import pandas as pd
import os, json
import argparse
import sys
for path in sys.path:
    if path.endswith("/eval"):
        sys.path.append(path.replace("/eval", "/"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

def load_metrics_from_directory(base_dir, dataset):
    all_data = []  # List to store all rows of data
    l2_metrics = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(os.path.join(base_dir, dataset)):
        for file in files:
            if file == "mle_model_metrics.json":
                file_path = os.path.join(root, file)
                parts = file_path.split(os.sep)
                head = parts[-4]
                gamma = parts[-3]
                freq = parts[-2] 
                with open(file_path, "r") as json_file:
                    metrics = json.load(json_file)
                    # Loop through each seed's results in the file
                    for seed, result in metrics.items():
                        # Append a row of data (with frequency and seed) to the all_data list
                        all_data.append({
                            "dataset": dataset,
                            "head": head,
                            "gamma": gamma,
                            "freqs": freq,
                            "seed": seed,
                            "KL divergence": result.get("KL divergence", None),
                            "Perplexity": result.get("Perplexity", None)
                        })

    #Convert all_data into a DataFrame
    df = pd.DataFrame(all_data)
    return df

def aggregate(output_dir, dataset_list, verbose=True):
    agg_data = []
    for dataset in dataset_list:
        df = load_metrics_from_directory(output_dir, dataset)

        # Group by 'freqs', 'gamma', and 'architecture' to aggregate the metrics across seeds
        aggregation = df.groupby(['dataset', 'freqs', 'gamma', 'head']).agg(
            kl_mean=('KL divergence', 'mean'),
            kl_std=('KL divergence', 'std'),
            perplexity_mean=('Perplexity', 'mean'),
            perplexity_std=('Perplexity', 'std')
        ).reset_index()

        result_df = aggregation.copy()
        result_df['freqs'] = pd.to_numeric(result_df['freqs'], errors='coerce') 
        result_df = result_df.sort_values(by=['freqs', 'gamma'])
        result_df['KL divergence'] = result_df['kl_mean'].round(3).astype(str) + r' $\pm$ ' + result_df['kl_std'].round(3).astype(str)
        result_df['Perplexity'] = result_df['perplexity_mean'].round(3).astype(str) + r' $\pm$ ' + result_df['perplexity_std'].round(3).astype(str)
       
        if verbose:
            print(result_df.to_string(index=False))

            # Find the model with the lowest KL divergence mean and round to 3 decimal places
            best_kl_model = result_df.loc[result_df['kl_mean'].idxmin()]
            best_kl_model_rounded = best_kl_model.round(3)
            print("\nBest model based on KL divergence (rounded):")
            print(best_kl_model_rounded)

            # Find the model with the lowest Perplexity mean and round to 3 decimal places
            best_perp_model = result_df.loc[result_df['perplexity_mean'].idxmin()]
            best_perp_model_rounded = best_perp_model.round(3)
            print("\nBest model based on Perplexity (rounded):")
            print(best_perp_model_rounded)

           

        # ---- Extracting Required Lists for gamma = 0 ----

        # Filter the data for gamma = 0 and freqs > 0
        filtered_df_gamma0 = result_df[(result_df['gamma'] == '0.0') & (result_df['freqs'] != 0)].sort_values(by='freqs')

        # Create the data dictionary
        data = {
            "KL Divergence": {
                "gamma0": [
                    filtered_df_gamma0['kl_mean'].round(3).tolist(),
                    filtered_df_gamma0['kl_std'].round(3).tolist()
                ],
            },
            "Perplexity": {
                "gamma0": [
                    filtered_df_gamma0['perplexity_mean'].round(3).tolist(),
                    filtered_df_gamma0['perplexity_std'].round(3).tolist()
                ],
            }
        }

        # ---- Extracting Required Lists for gamma = 1e-6 ----

        # Filter the data for gamma = 1e-6 and freqs > 0
        filtered_df_gamma1 = result_df[(result_df['gamma'] == '1e-06') & (result_df['freqs'] != 0)].sort_values(by='freqs')

        data["KL Divergence"]["gamma1"] = [
            filtered_df_gamma1['kl_mean'].round(3).tolist(),
            filtered_df_gamma1['kl_std'].round(3).tolist()
        ]

        data["Perplexity"]["gamma1"] = [
            filtered_df_gamma1['perplexity_mean'].round(3).tolist(),
            filtered_df_gamma1['perplexity_std'].round(3).tolist()
        ]


        gmm = result_df[(result_df['head'] == 'gmm-mle')]
        gmm_values = {"KL Divergence" : float(gmm['kl_mean'].iloc[0]), "Perplexity": float(gmm['perplexity_mean'].iloc[0])}
        gmm_stds = {"KL Divergence" : float(gmm['kl_std'].iloc[0]), "Perplexity": float(gmm['perplexity_std'].iloc[0])}
        
        agg_data.append((data, gmm_values, gmm_stds))

    return agg_data
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse input arguments.")
        
    # Adding arguments
    parser.add_argument('--dir', type=str, required=True, 
                            help='Specify output dir (string)')
    parser.add_argument('--datasets', nargs='+', required=True, 
                            help='Datasets list')
        
    # Parsing arguments
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    aggregate(args.dir, args.datasets)