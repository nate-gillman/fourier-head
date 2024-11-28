def get_t5_scaling_configs():
    # Base configuration that will be common across all models
    base_config = {
        "architectures": ["T5ForConditionalGeneration"],
        "classifier_dropout": 0.0,
        "decoder_start_token_id": 0,
        "dense_act_fn": "relu",
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "relu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "is_gated_act": False,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "n_positions": 512,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "torch_dtype": "float32",
        "transformers_version": "4.42.3",
        "use_cache": True,
        "vocab_size": 32128
    }
    
    # Define scaling configurations
    # Following DeepNarrow strategy, we prioritize depth (num_layers) scaling
    # Key 4 is tiny, Key 9 is mini
    configs = {
        0: {
            "d_ff": 512,          # Half of tiny
            "d_kv": 32,           # Half of tiny
            "d_model": 128,       # Half of tiny
            "num_heads": 2,       # Half of tiny
            "num_decoder_layers": 2,  # Half of tiny
            "num_layers": 2,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 1246848
        },
        1: {
            "d_ff": 750,
            "d_kv": 48,
            "d_model": 186,
            "num_heads": 3,
            "num_decoder_layers": 2,
            "num_layers": 2,
            "_name_or_path": "t5-efficient-nano-1",
            "num_params" : 2523096
        },
        2: {
            "d_ff": 850,
            "d_kv": 56,
            "d_model": 224,
            "num_heads": 4,
            "num_decoder_layers": 3,
            "num_layers": 3,
            "_name_or_path": "t5-efficient-nano-2",
            "num_params" : 5012704
        },
        3: {
            "d_ff": 1024,
            "d_kv": 64,
            "d_model": 288,
            "num_heads": 5,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-small-0",
            "num_params" : 10328576
        },
        # Mini model (index 9)
        4: {
            "d_ff": 1536,
            "d_kv": 64,
            "d_model": 384,
            "num_heads": 8,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "google/t5-efficient-mini",
            "num_params" : 20456192
        }
    }
    
    # Merge base config with specific configs
    return {k: {**base_config, **v} for k, v in configs.items()}

t5_scaling_configs = get_t5_scaling_configs()