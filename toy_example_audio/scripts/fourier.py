"""
This is an example of how one can run a single run of a fourier head.
"""
import wandb

from experiment.training import start_training
from config import AUDIO_REPO_ROOT

if __name__ == "__main__":
    # run with some number of frequencies
    fourier_num_frequencies = 2
    test_run_dir = AUDIO_REPO_ROOT / "output" / str(fourier_num_frequencies)

    wandb_project = None  # "fourier_audio"
    if wandb_project:
        wandb.init()

    start_training(
        wandb_project=wandb_project,
        output_dir=test_run_dir,
        fourier_num_frequencies=fourier_num_frequencies,
        dataset_split_style="odds_evens",
    )
