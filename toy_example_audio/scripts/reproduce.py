from config import AUDIO_REPO_ROOT
from experiment.training import start_training
from experiment.graphs import create_graph

if __name__ == "__main__":
    # run the training of each model configuration and save results on disk
    for fourier_num_frequencies in range(3):
        # run the fourier model with 0-2 frequencies, 0 = linear
        test_run_dir = AUDIO_REPO_ROOT / "output" / str(fourier_num_frequencies)
        start_training(
            wandb_project=None,
            output_dir=test_run_dir,
            fourier_num_frequencies=fourier_num_frequencies,
            dataset_bin_size=5,
            dataset_split_style="standard",
        )

    # save the graph for figure 1 to disk in the output folder
    create_graph()
