echo "exps_linear"

num_steps=500000

for seed in 123 231 312 42
do
    echo "num_steps = $num_steps"
    echo "seed=$seed"

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
        --save_path ./output/linear
done