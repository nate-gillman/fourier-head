SCALE_CONFIGS = {
    0: {"name": "Nano",     "n_layer": 2,   "n_head": 2,    "n_embd": 32,    "M_log_params": 0.289},    # ~0.2895M params
    1: {"name": "Micro",    "n_layer": 2,   "n_head": 2,    "n_embd": 48,    "M_log_params": 0.4137},    # ~0.4137M params
    2: {"name": "Tiny",     "n_layer": 2,   "n_head": 4,    "n_embd": 64,    "M_log_params": 0.5502},    # ~0.5502M params
    3: {"name": "Mini",     "n_layer": 3,   "n_head": 4,    "n_embd": 64,    "M_log_params": 0.6002},    # ~0.6002M params
    4: {"name": "XSmall",   "n_layer": 3,   "n_head": 4,    "n_embd": 96,    "M_log_params": 0.9719},    # ~0.9719M params
    5: {"name": "Small-A",  "n_layer": 4,   "n_head": 4,    "n_embd": 96,    "M_log_params": 1.084},    # ~1.084M params
    6: {"name": "Small-B",  "n_layer": 4,   "n_head": 6,    "n_embd": 96,    "M_log_params": 1.084},    # ~1.084M params
    7: {"name": "Small-C",  "n_layer": 4,   "n_head": 6,    "n_embd": 120,   "M_log_params": 1.473},   # ~1.473M params
    8: {"name": "Small-D",  "n_layer": 5,   "n_head": 6,    "n_embd": 120,   "M_log_params": 1.648},   # ~1.648M params
    9: {"name": "Small-E",  "n_layer": 5,   "n_head": 8,    "n_embd": 128,   "M_log_params": 1.814},   # ~1.814M params
    10: {"name": "Base",    "n_layer": 6,   "n_head": 8,    "n_embd": 128,   "M_log_params": 2.012},   # ~2.012M params
}