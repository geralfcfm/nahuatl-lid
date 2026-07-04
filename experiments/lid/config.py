from __future__ import annotations

import random
import numpy as np
import torch

SEED = 42
SAMPLE_RATE = 16000
DURATION = 3
SAMPLES_PER_CHUNK = SAMPLE_RATE * DURATION  # 48000
N_MELS = 64
N_FFT = 1024
HOP = 512
F_MIN = 50
K_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 32
LR = 5e-4

BANDS = {
    "wideband_16k":   {"f_max": 8000, "lowpass_hz": None},
    "bandrestrict_4k": {"f_max": 4000, "lowpass_hz": None},
    "lowpass_4k":     {"f_max": 8000, "lowpass_hz": 4000},
}
NORMS = ("none", "instance")
CONFIGS = [(b, n) for b in BANDS for n in NORMS]

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # GPU determinism: seed-stable; note cuDNN LSTM backward is not bitwise-deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
