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
        # Models smaller than tiny
        0: {
            "d_ff": 200,          # Half of tiny
            "d_kv": 12,           # Half of tiny
            "d_model": 32,       # Half of tiny
            "num_heads": 1,       # Half of tiny
            "num_decoder_layers": 1,  # Half of tiny
            "num_layers": 1,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 161568
        },
        1: {
            "d_ff": 250,          # Half of tiny
            "d_kv": 16,           # Half of tiny
            "d_model": 48,       # Half of tiny
            "num_heads": 2,       # Half of tiny
            "num_decoder_layers": 1,  # Half of tiny
            "num_layers": 1,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 263504
        },
        2: {
            "d_ff": 300,          # Half of tiny
            "d_kv": 24,           # Half of tiny
            "d_model": 64,       # Half of tiny
            "num_heads": 2,       # Half of tiny
            "num_decoder_layers": 2,  # Half of tiny
            "num_layers": 2,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 490368
        },
        3: {
            "d_ff": 400,          # Half of tiny
            "d_kv": 24,           # Half of tiny
            "d_model": 100,       # Half of tiny
            "num_heads": 2,       # Half of tiny
            "num_decoder_layers": 2,  # Half of tiny
            "num_layers": 2,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 846128
        },
        4: {
            "d_ff": 512,          # Half of tiny
            "d_kv": 32,           # Half of tiny
            "d_model": 128,       # Half of tiny
            "num_heads": 2,       # Half of tiny
            "num_decoder_layers": 2,  # Half of tiny
            "num_layers": 2,      # Half of tiny
            "_name_or_path": "t5-efficient-nano-0",
            "num_params" : 1246848
        },
        5: {
            "d_ff": 768,
            "d_kv": 48,
            "d_model": 192,
            "num_heads": 3,
            "num_decoder_layers": 3,
            "num_layers": 3,
            "_name_or_path": "t5-efficient-nano-1",
            "num_params" : 3554688
        },
        6: {
            "d_ff": 896,
            "d_kv": 56,
            "d_model": 224,
            "num_heads": 4,
            "num_decoder_layers": 3,
            "num_layers": 3,
            "_name_or_path": "t5-efficient-nano-2",
            "num_params" : 5136352
        },
        7: {
            "d_ff": 960,
            "d_kv": 60,
            "d_model": 240,
            "num_heads": 4,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-nano-3",
            "num_params" : 7439776
        },
        # Tiny model (index 4)
        8: {
            "d_ff": 1024,
            "d_kv": 64,
            "d_model": 256,
            "num_heads": 4,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "google/t5-efficient-tiny",
            "num_params" : 8394496
        },
        # Models between tiny and mini
        9: {
            "d_ff": 1152,
            "d_kv": 64,
            "d_model": 288,
            "num_heads": 6,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-small-0",
            "num_params" : 11803200 
        },
        10: {
            "d_ff": 1280,
            "d_kv": 64,
            "d_model": 320,
            "num_heads": 6,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-small-1",
            "num_params" : 13769984
        },
        11: {
            "d_ff": 1408,
            "d_kv": 64,
            "d_model": 352,
            "num_heads": 7,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-small-2",
            "num_params" : 16949248
        },
        12: {
            "d_ff": 1472,
            "d_kv": 64,
            "d_model": 368,
            "num_heads": 8,
            "num_decoder_layers": 4,
            "num_layers": 4,
            "_name_or_path": "t5-efficient-small-3",
            "num_params" : 19227040 
        },
        # Mini model (index 9)
        13: {
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