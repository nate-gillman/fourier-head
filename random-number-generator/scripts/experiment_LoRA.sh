#!/bin/bash
set -e

# parse command line arguments
NUM_IN_CONTEXT_SAMPLES_PER_PROMPT=$1
DATA_DIR=$2
NUM_EPOCHS=$3
NUM_FREQUENCIES=$4  # options: [0 = linear, >1 = Fourier]
SEED=$5
echo -e "\n\nData Directory:                               $DATA_DIR"
echo "Number of Epochs:                             $NUM_EPOCHS"
echo "Number of in-context samples per prompt:      $NUM_IN_CONTEXT_SAMPLES_PER_PROMPT"
echo "Number of Frequencies (0=linear, >0=Fourier): $NUM_FREQUENCIES"
echo -e "Random Seed:                                  $SEED\n\n"

IS_LORA_MODEL=True          # options: [True, False]
VOCAB_SIZE=200              # options: integers

SAVE_DIR=output/$(printf "%02d" $NUM_IN_CONTEXT_SAMPLES_PER_PROMPT)_in_context_samples_per_prompt/epochs-$NUM_EPOCHS-freqs-$NUM_FREQUENCIES-seed-$SEED

mkdir -p $SAVE_DIR

# fine-tune the model
if [[ "$IS_LORA_MODEL" == "True" ]]; then
    python scripts/train_LoRA.py \
        --output_dir $SAVE_DIR \
        --num_frequencies $NUM_FREQUENCIES \
        --vocab_size $VOCAB_SIZE \
        --num_epochs $NUM_EPOCHS \
        --seed $SEED \
        --data_dir $DATA_DIR
fi

# (in domain) inference using trained model
python scripts/inference.py \
    --data_dir $DATA_DIR \
    --output_dir $SAVE_DIR \
    --test_split in_domain \
    --is_LoRA_model $IS_LORA_MODEL

# (in domain) compute metrics using inference output
python scripts/eval.py \
    --output_dir $SAVE_DIR \
    --test_split in_domain
