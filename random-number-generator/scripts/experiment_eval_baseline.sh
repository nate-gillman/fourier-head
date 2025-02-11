#!/bin/bash
set -e

# parse command line arguments
NUM_IN_CONTEXT_SAMPLES_PER_PROMPT=$1
DATA_DIR=$2
SEED=$3
echo -e "\n\nData Directory:                               $DATA_DIR"
echo "Number of in-context samples per prompt:      $NUM_IN_CONTEXT_SAMPLES_PER_PROMPT"
echo -e "Random Seed:                                  $SEED\n\n"

IS_LORA_MODEL=False          # options: [True, False]

SAVE_DIR=output/$(printf "%02d" $NUM_IN_CONTEXT_SAMPLES_PER_PROMPT)_in_context_samples_per_prompt/original-model-baseline-seed-$SEED

mkdir -p $SAVE_DIR

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
