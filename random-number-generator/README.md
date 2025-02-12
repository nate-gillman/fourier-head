# Toy Example: Are LLMs Random Number Generators?

## Step 1: Environment setup


### Step 1a: build conda environment

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

### Step 1b: download the llama model

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

## Step 2: build the dataset

We have already committed our training data to the `data` folder.
If you wish to re-create it, then you can run the following script.

```bash
train_set_size=256
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/build_dataset_json.py \
        --num_in_context_samples_per_prompt $num_in_context_samples_per_prompt \
        --train_set_size $train_set_size
done
```

## Step 3: evaluate the baseline

This whole loop can run in less than 6 hours on a 3090 gpu.
This doesn't require any fine-tuning, only inference on the pretrained model.

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

## Step 4: LoRA fine-tune, run inference and eval scripts

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
Note that, for a fixed frequency, and a fixed number of in-context samples per prompt, it takes between 45 minutes and 2 hours to run the inner loop over the 10 seeds.

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



## Step 5: graph the results

### Step 5a: aggregate metrics across seeds, in preparation for graphing

Once the training experiments are done, the following script should be run to aggregate metrics into json files.
This takes less than a minute and doesn't need GPU.

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/aggregate_metrics_across_seeds.py \
        output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt
done
```

### Step 5b: x_axis = num_freqs, y_axis = tvd, or num_unique_samples

Graphing the total variation distance and num_unique_sample metrics as a function of frequencies:

```bash
metrics=(tvd num_unique_samples)
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)

for metric in "${metrics[@]}"; do
    for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
        python scripts/graph_metrics_varying_frequencies.py \
            --metric ${metric} \
            --input_dir output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
            --output_dir output/graphs/num_in_context_samples_per_prompt_0${num_in_context_samples_per_prompt} \
            --max_num_freqs 12
    done
done
```

### Step 5c: x_axis = num_in_context_samples_per_prompt, y_axis = tvd, or num_unique_samples

Graphing the total variation distance and num_unique_sample metrics as a function of num_in_context_samples_per_prompt:

```bash 
metrics=(tvd num_unique_samples)
freqs=(1 2 3 4 5 6 7 8 9 10 11 12)

for metric in "${metrics[@]}"; do
    for freq in "${freqs[@]}"; do
        python scripts/graph_metrics_varying_in_context_samples.py \
            --metric ${metric} \
            --input_dir output \
            --output_dir output/graphs/num_freqs_${freq} \
            --freq $freq \
            --max_num_in_context_samples_per_prompt 9
    done
done
```


### Step 5d: graph the learned empirical distributions

This will graph the true distribution, against the learned distribution (obtained via sampling and histogram binning).

```bash
nums_in_context_samples_per_prompt=(0 1 2 3 4 5 6 7 8 9)
for num_in_context_samples_per_prompt in "${nums_in_context_samples_per_prompt[@]}"; do
    python scripts/graph_predicted_distributions.py \
            --input_dir output/0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt \
            --output_dir output/graphs/pred_distributions_0${num_in_context_samples_per_prompt}_in_context_samples_per_prompt
done
```