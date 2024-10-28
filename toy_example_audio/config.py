from pathlib import Path

# assuming this file is located in
# ./toy_example_audio/config.py

# AUDIO_REPO_ROOT is the directory containing the audio toy example code (toy_example_audio)
AUDIO_REPO_ROOT = Path(__file__).parent
OUTPUT_FOLDER = AUDIO_REPO_ROOT / "output"
SWEEP_ARTIFACT_FOLDER = OUTPUT_FOLDER / "wandb_sweeps"
GRAPH_OUTPUT_FOLDER = OUTPUT_FOLDER / "graphs"

# Set as desired, this only affects the audio toy example.
PROJECT_SEED = 42