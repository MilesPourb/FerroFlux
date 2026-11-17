"""Polite Login-Node Resource Throttling & Deterministic Seeding (`login06.anvil.rcac.purdue.edu`)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def enforce_login_node_politeness(max_threads: int = 6, seed: int = 20261116) -> dict[str, object]:
    """Throttle CPU thread pools on login nodes while allowing full thread scaling on SLURM compute nodes."""
    nodename = os.uname().nodename.lower()
    on_slurm_compute = ("SLURM_JOB_ID" in os.environ) and ("login" not in nodename)
    cap = 128 if on_slurm_compute else 8
    threads = max(1, min(int(max_threads), cap))
    for env_key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_key] = str(threads)

    niced = None
    if not on_slurm_compute:
        try:
            cur_nice = os.nice(0)
            if cur_nice < 10:
                niced = os.nice(10 - cur_nice)
            else:
                niced = cur_nice
        except OSError:
            niced = None

    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return {
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "nice_level": niced,
        "max_threads": threads,
        "torch_num_threads": torch.get_num_threads(),
        "seed": seed,
    }


def get_peak_rss_mb() -> float:
    """Read Linux `/proc/self/status` `VmHWM` (high-water mark RSS) in MiB."""
    status_p = Path("/proc/self/status")
    if status_p.is_file():
        for line in status_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / 1024.0, 2)
    return 0.0
