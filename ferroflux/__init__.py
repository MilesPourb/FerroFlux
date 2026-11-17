"""FerroFlux-FM: Mechanistic Neural-Biophysical Foundation Model for Ferroptosis."""

import os
import socket
from pathlib import Path

if "SLURM_JOB_ID" not in os.environ and socket.gethostname().lower().startswith("login"):
    for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_env_k, "4")
    try:
        import torch as _torch

        if _torch.get_num_threads() > 8:
            _torch.set_num_threads(4)
    except Exception:
        pass

__version__ = "0.3.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

__all__ = [
    "__version__",
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
]
