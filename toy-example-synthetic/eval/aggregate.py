import pandas as pd
import os, json
import argparse
import sys
for path in sys.path:
    if path.endswith("/eval"):
        sys.path.append(path.replace("/eval", "/"))

import numpy as np

def load_metrics_from_directory(base_dir, dataset):
    all_data = []  # List to store all rows of data
    l2_metrics = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(os.path.join(base_dir, dataset)):
        for file in files:
            if file == "model_metrics.json":
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
                            "MSE": result.get("MSE", None),
                            "MSE_argmax": result.get("MSE_argmax", None)
                        })

                smoothness_file = os.path.join(root, "smoothness_dict.json")
                if not os.path.isfile(smoothness_file):
                    from eval.compute_smoothness_dict import compute_and_save_smoothness_dict_in_dir
                    compute_and_save_smoothness_dict_in_dir(root)

                with open(smoothness_file, "r") as json_file:
                    data = json.load(json_file)
                    smoothness = data['L2']
                    l2_metrics.append({
                        "dataset": dataset,
                        "head": head,
                        "gamma": gamma,
                        "freqs": freq,
                        "L2_mean": smoothness['mean'],
                        "L2_std": smoothness['std'],
                                })

    #Convert all_data into a DataFrame
    df = pd.DataFrame(all_data)
    l2_df = pd.DataFrame(l2_metrics)
    return df, l2_df

def aggregate(output_dir, dataset_list, verbose=True):
    agg_data = []
    for dataset in dataset_list:
        df, l2_df = load_metrics_from_directory(output_dir, dataset)

        # Group by 'freqs', 'gamma', and 'architecture' to aggregate the metrics across seeds
        aggregation = df.groupby(['dataset', 'freqs', 'gamma', 'head']).agg(
            kl_mean=('KL divergence', 'mean'),
            kl_std=('KL divergence', 'std'),
            mse_mean=('MSE', 'mean'),
            mse_std=('MSE', 'std'),
            mse_max_mean=('MSE_argmax', 'mean'),
            mse_max_std=('MSE_argmax', 'std')
        ).reset_index()

        result_df = pd.merge(aggregation, l2_df, on=["dataset", "head", "gamma", "freqs"], how="outer")
        result_df['freqs'] = pd.to_numeric(result_df['freqs'], errors='coerce') 
        result_df = result_df.sort_values(by=['freqs', 'gamma'])
        result_df['KL divergence'] = result_df['kl_mean'].round(3).astype(str) + r' $\pm$ ' + result_df['kl_std'].round(3).astype(str)
        result_df['MSE'] = result_df['mse_mean'].round(3).astype(str) + r' $\pm$ ' + result_df['mse_std'].round(3).astype(str)
        result_df['MSE_argmax'] = result_df['mse_max_mean'].round(3).astype(str) + r' $\pm$ ' + result_df['mse_max_std'].round(3).astype(str)
        if verbose:
            print(result_df.to_string(index=False))

            # Find the model with the lowest KL divergence mean and round to 3 decimal places
            best_kl_model = result_df.loc[result_df['kl_mean'].idxmin()]
            best_kl_model_rounded = best_kl_model.round(3)
            print("\nBest model based on KL divergence (rounded):")
            print(best_kl_model_rounded)

            # Find the model with the lowest MSE mean and round to 3 decimal places
            best_mse_model = result_df.loc[result_df['mse_mean'].idxmin()]
            best_mse_model_rounded = best_mse_model.round(3)
            print("\nBest model based on MSE (rounded):")
            print(best_mse_model_rounded)

            # Find the model with the lowest MSE mean (argmax) and round to 3 decimal places
            best_mse_model = result_df.loc[result_df['mse_max_mean'].idxmin()]
            best_mse_model_rounded = best_mse_model.round(3)
            print("\nBest model based on MSE_argmax (rounded):")
            print(best_mse_model_rounded)


            # Find the model with the lowest L2_mean and round to 4 decimal places
            best_l2_model = result_df.loc[result_df['L2_mean'].idxmin()]
            best_l2_model_rounded = best_l2_model.round(4)
            print("\nBest model based on L2 (rounded):")
            print(best_l2_model_rounded)


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
            "MSE": {
                "gamma0": [
                    filtered_df_gamma0['mse_mean'].round(3).tolist(),
                    filtered_df_gamma0['mse_std'].round(3).tolist()
                ],
            },
            "Smoothness": {
                "gamma0": [
                    filtered_df_gamma0['L2_mean'].round(4).tolist(),
                    filtered_df_gamma0['L2_std'].round(4).tolist()
                ],
            },
        }

        # ---- Extracting Required Lists for gamma = 1e-6 ----

        # Filter the data for gamma = 1e-6 and freqs > 0
        filtered_df_gamma1 = result_df[(result_df['gamma'] == '1e-06') & (result_df['freqs'] != 0)].sort_values(by='freqs')

        data["KL Divergence"]["gamma1"] = [
            filtered_df_gamma1['kl_mean'].round(3).tolist(),
            filtered_df_gamma1['kl_std'].round(3).tolist()
        ]
        data["MSE"]["gamma1"] = [
            filtered_df_gamma1['mse_mean'].round(3).tolist(),
            filtered_df_gamma1['mse_std'].round(3).tolist()
        ]
        data["Smoothness"]["gamma1"] = [
            filtered_df_gamma1['L2_mean'].round(4).tolist(),
            filtered_df_gamma1['L2_std'].round(4).tolist()
        ]
        
        linear = result_df[(result_df['head'] == 'linear')]
        baseline_values = {"KL Divergence" : float(linear['kl_mean'].iloc[0]), "MSE": float(linear['mse_mean'].iloc[0]), "Smoothness": float(linear['L2_mean'].iloc[0])}
        baseline_stds = {"KL Divergence" : float(linear['kl_std'].iloc[0]), "MSE": float(linear['mse_std'].iloc[0]), "Smoothness": float(linear['L2_std'].iloc[0])}
        agg_data.append((data, baseline_values, baseline_stds))

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