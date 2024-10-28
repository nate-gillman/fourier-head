echo "exps_fourier"

num_steps=500000
fourier_frequencies=$1

for seed in 123 231 312 42
do
    echo "num_steps = $num_steps"
    echo "seed=$seed"
    echo "fourier_frequencies = $fourier_frequencies"

    python run_dt_atari.py \
        --seed $seed \
        --context_length 30 \
        --epochs 5 \
        --model_type 'reward_conditioned' \
        --num_steps $num_steps \
        --num_buffers 50 \
        --game 'Seaquest' \
        --batch_size 128 \
        --data_dir_prefix ./dataset/ \
        --fourier_frequencies $fourier_frequencies \
        --save_path ./output/fourier_$fourier_frequencies
done