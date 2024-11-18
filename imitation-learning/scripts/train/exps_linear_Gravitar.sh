echo "exps_linear"

num_steps=500000
game_name=$1

for seed in 231
do
    echo "game_name=$game_name"
    echo "num_steps = $num_steps"
    echo "seed=$seed"

    python run_dt_atari.py \
        --seed $seed \
        --context_length 30 \
        --epochs 5 \
        --model_type 'reward_conditioned' \
        --num_steps $num_steps \
        --num_buffers 50 \
        --game "$game_name" \
        --batch_size 128 \
        --data_dir_prefix ./dataset/ \
        --save_path ./output/$game_name-v3/linear/
done