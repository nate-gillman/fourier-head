SCALE_CONFIGS = {
    0: {"name": "Nano",     "n_layer": 2,   "n_head": 2,    "n_embd": 32},    # ~0.2895M params
    1: {"name": "Micro",    "n_layer": 2,   "n_head": 2,    "n_embd": 48},    # ~0.4137M params
    2: {"name": "Tiny",     "n_layer": 2,   "n_head": 4,    "n_embd": 64},    # ~0.5502M params
    3: {"name": "Mini",     "n_layer": 3,   "n_head": 4,    "n_embd": 64},    # ~0.6002M params
    4: {"name": "XSmall",   "n_layer": 3,   "n_head": 4,    "n_embd": 96},    # ~0.9719M params
    5: {"name": "Small-A",  "n_layer": 4,   "n_head": 4,    "n_embd": 96},    # ~1.084M params
    6: {"name": "Small-B",  "n_layer": 4,   "n_head": 6,    "n_embd": 96},    # ~1.084M params
    7: {"name": "Small-C",  "n_layer": 4,   "n_head": 6,    "n_embd": 120},   # ~1.473M params
    8: {"name": "Small-D",  "n_layer": 5,   "n_head": 6,    "n_embd": 120},   # ~1.648M params
    9: {"name": "Small-E",  "n_layer": 5,   "n_head": 8,    "n_embd": 128},   # ~1.814M params
    10: {"name": "Base",    "n_layer": 6,   "n_head": 8,    "n_embd": 128},   # ~2.012M params
    11: {"name": "Medium-A","n_layer": 7,   "n_head": 8,    "n_embd": 192},   # ~4.309M params
    12: {"name": "Medium-B","n_layer": 8,   "n_head": 8,    "n_embd": 256},   # ~7.885M params
    13: {"name": "Medium-C","n_layer": 10,  "n_head": 10,   "n_embd": 320},   # ~14.27M params
    14: {"name": "Large-A", "n_layer": 12,  "n_head": 12,   "n_embd": 384},   # ~23.61M params
    15: {"name": "Large-B", "n_layer": 14,  "n_head": 14,   "n_embd": 448},   # ~36.48M params
    16: {"name": "XLarge-A","n_layer": 16,  "n_head": 16,   "n_embd": 512},   # ~53.49M params
    17: {"name": "XLarge-B","n_layer": 18,  "n_head": 18,   "n_embd": 576},   # ~75.23M params
    18: {"name": "2XLarge", "n_layer": 20,  "n_head": 20,   "n_embd": 640},   # ~102.3M params
}