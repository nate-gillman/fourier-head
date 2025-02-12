# Toy Example: Are LLMs Random Number Generators?

## Step 1: Environment setup


### Conda env

First, create conda environment.

```bash
# CONDA_ENV_DIR=/path/to/where/you/want/to/store/your/env
CONDA_ENV_DIR=./llama-fourier-head-env
conda create -p $CONDA_ENV_DIR python=3.11
conda activate $CONDA_ENV_DIR

# install torch, verify it was installed correctly
pip install --prefix=$CONDA_ENV_DIR torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(5); a = a.to("cuda:0"); print(a)'

pip install --prefix=$CONDA_ENV_DIR llama-recipes
pip install --prefix=$CONDA_ENV_DIR ipywidgets
pip install --prefix=$CONDA_ENV_DIR wandb
```

### Download the models

You'll need to sign into HuggingFace, and have access to the Llama models.
Inside the python interpreter, do the following:

```python
import huggingface_hub
huggingface_hub.login()
```

The following script downloads the Llama model we'll use, and tests the download by running inference on a dummy example.

```bash
python scripts/download_llama.py
```

If the models download, and eventually training begins, then it worked!

## Step 2: build the dataset

You don't need to run this, as we already committed our data to the `data` folder.
But if you want to re-run the script, go for it.
This gives in-context examples, and doesn't say describe in english the nature of the distribution, opting instead to only provide samples from the distribution.

```bash
train_set_size=256
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/build_dataset_json.py \
        --num_in_context_samples_per_prompt $num_in_context_samples_per_prompt \
        --train_set_size $train_set_size
done
```

## Step 3: evaluate the baseline (doesn't require any fine-tuning)

This whole loop can run in less than 6 hours on a 3090 gpu.

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
seeds=(42 43 44 45 46 47 48 49 50 51)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    data_dir=data/$(printf "%02d" $num_in_context_samples_per_prompt)_in_context_samples
    for seed in "${seeds[@]}"; do
        sh scripts/experiment_eval_baseline.sh $num_in_context_samples_per_prompt $data_dir $seed
    done
done
```

## Step 4: fine-tune, run inference and eval scripts

LoRA train the linear baseline, then evaluate the trained model.
This whole loop can run in less than 20 hours on an a6000 gpu.

```bash
num_epochs=16
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
num_freqs=0
seeds=(42 43 44 45 46 47 48 49 50 51)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    data_dir=data/$(printf "%02d" $num_in_context_samples_per_prompt)_in_context_samples
    for seed in "${seeds[@]}"; do
        sh scripts/experiment_LoRA.sh $num_in_context_samples_per_prompt $data_dir $num_epochs $num_freqs $seed
    done
done
```

LoRA train the fourier model, and then evaluate it.
Each inner loop over the 10 seeds runs in less than 2 hours on a 3090 gpu.

```bash
num_epochs=16
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
nums_freqs=(1 2 3 4 5 6 7 8 9 10 11 12)
seeds=(42 43 44 45 46 47 48 49 50 51)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    for num_freqs in "${nums_freqs[@]}"; do
        data_dir=data/$(printf "%02d" $num_in_context_samples_per_prompt)_in_context_samples
        for seed in "${seeds[@]}"; do
            sh scripts/experiment_LoRA.sh $num_in_context_samples_per_prompt $data_dir $num_epochs $num_freqs $seed
        done
    done
done
```

Note that, for a fixed frequency, and a fixed number of in-context samples per prompt, it takes approximately one hour to run the inner loop over all 10 seeds.

## Step 5: aggregate metrics across seeds, in preparation for graphing

Once the training experiments are done, we can indeed aggregate metrics while the experiments are still in progress.
This takes less than a minute and doesn't need GPU.

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/aggregate_metrics_across_seeds.py \
        output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt
done
```

## Step 6: graph the results

### x_axis = num_freqs, y_axis = {metric}

Graphing the total variation distance metric as a function of frequencies:

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/graph_metrics.py \
        --metric tvd \
        --input_dir output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
        --output_dir scripts/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
        --max_num_freqs 12
done
```

Graphing the num_unique_samples metric as a function of frequencies:

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/graph_metrics.py \
        --metric num_unique_samples \
            --input_dir output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
            --output_dir scripts/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
            --max_num_freqs 12
done
```

## x_axis = num_in_context_samples_per_prompt, y_axis = {metric}

Graphing the TVD metric as a function of num_in_context_samples_per_prompt:

```bash 
python scripts/graph_metrics_varying_in_context_samples.py \
    --metric tvd \
    --input_dir output \
    --output_dir scripts/graph_metrics_varying_in_context_samples \
    --freqs_to_graph 1,2,3,4,5,6,7,8,9,10,11,12 \
    --max_num_in_context_samples_per_prompt 9
```

Graphing the num_unique_samples metric as a function of num_in_context_samples_per_prompt:

```bash
python scripts/graph_metrics_varying_in_context_samples.py \
    --metric num_unique_samples \
    --input_dir output \
    --output_dir scripts/graph_metrics_varying_in_context_samples \
    --freqs_to_graph 1,2,3,4,5,6,7,8,9,10,11,12 \
    --max_num_in_context_samples_per_prompt 9
```


## STEP 9: graph predicted distributions 

This will graph the true distribution, against the learned distribution (obtained via sampling and histogram binning).

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/graph_predicted_distributions.py \
            --input_dir output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
            --output_dir scripts/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt
done
```