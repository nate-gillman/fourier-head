echo "exps_linear"

num_steps=500
game_name=$1
scale=18

for seed in 123 231 312 42
do
    echo "game_name=$game_name"
    echo "num_steps = $num_steps"
    echo "seed=$seed"
    echo "scale=$scale"

    python run_dt_atari_copy.py \
        --seed $seed \
        --context_length 30 \
        --epochs 5 \
        --model_type 'reward_conditioned' \
        --num_steps $num_steps \
        --num_buffers 50 \
        --game "$game_name" \
        --batch_size 128 \
        --data_dir_prefix ./dataset/ \
        --save_path ./output/$game_name-$scale/linear/ \
        --scale $scale
done