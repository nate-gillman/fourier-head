### Fourier Audio Toy Example

#### Creating Environment

From within this folder (first `cd toy_example_audio`), run:
```bash
conda env create -f environment.yaml
```

And activate the environment with:
```bash
conda activate fourier_head_audio
```

If any issues arise when installing from the `yaml` file, one can recreate it from scratch by installing [PyTorch](https://pytorch.org/get-started/locally/) and then installing the below:
```bash
pip install transformers==4.42.3
pip install datasets==2.20.0
pip install evaluate==0.4.3
pip install scikit-learn==1.5.1
pip install scipy==1.14.0
pip install wandb
pip install soundfile
pip install librosa
```

#### Running the Experiment

To run the experiment, from the root of the repository run:
```bash
python scripts/reproduce.py
```

This will run a linear head, fourier head with one frequency, and fourier head with two frequencies on the same task. The run data such as
checkpoints, configuration, and evaluation results will be saved to `output/`. The directories will be named by the
number of frequencies given, with `0` being linear (baseline), `1` being one frequency, and `2` being two frequencies. 

At the end of the script, the output folder should look _approximately_ like this:
```
toy_example_audio/
    output/
        0/
            run-0/
                ast_classifier/
                    checkpoint-x/
                    checkpoint-y/
                0-eval_pred_label_ids... .npy
                0-eval_pred_predictions... .npy
                1-eval_pred_label_ids... .npy
                1-eval_pred_predictions... .npy
                info.json
        1/
            run-0/
                ast_classifier/
                    checkpoint-x/
                    checkpoint-y/
                0-eval_pred_label_ids... .npy
                0-eval_pred_predictions... .npy
                1-eval_pred_label_ids... .npy
                1-eval_pred_predictions... .npy
                info.json
        2/
            run-0/
                ast_classifier/
                    checkpoint-x/
                    checkpoint-y/
                0-eval_pred_label_ids... .npy
                0-eval_pred_predictions... .npy
                1-eval_pred_label_ids... .npy
                1-eval_pred_predictions... .npy
                info.json
```

The script will download a [dataset](https://huggingface.co/datasets/meganwei/syntheory) from HuggingFace Hub, requiring about 1.2 GB of disk space. More information and the code that generates the SynTheory dataset is here: [SynTheory GitHub](https://github.com/brown-palm/syntheory)

When all of the models finish training and evaluating, the graph we show in the paper will be written to disk as a `.png` file. Check the folder:
```bash
toy_example_audio/output/graphs/...
```

#### Running Hyperparameter Sweeps

We provide some helper scripts to run a hyperparameter sweep using [Weights & Biases](https://wandb.ai/site/) (`wandb`) and [slurm](https://en.wikipedia.org/wiki/Slurm_Workload_Manager). Neither are required to run the experiments or sweeps, but we provide them for convenience. Admittedly, these scripts are (highly) likely to require some tweaking to fit one's computing infrastructure. They are not a general solution and will not 'just work' as is.

After adjusting the settings and reading the code, to start the sweep, run:
```bash
sbatch run_sweep.sh
```
