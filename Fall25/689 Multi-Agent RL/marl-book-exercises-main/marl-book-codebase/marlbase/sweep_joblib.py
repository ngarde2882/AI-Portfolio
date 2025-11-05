"""
Parallel sweep runner for MARL experiments (Hydra-free Joblib version).
Runs combinations of hyperparameters via joblib.Parallel.
Each run executes run.py with its own overrides and output directory.
"""

import os
import sys
import subprocess
from joblib import Parallel, delayed
from itertools import product
from pathlib import Path


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
PYTHON = sys.executable  # your current env Python
RUN_SCRIPT = "run.py"    # path to main training entry

# Sweep parameters
ENTROPY_COEFS = [.01,.05,.001]
LRS = [1e-4,3e-4]
GAMMAS = [0.99]
SEED = 0

ENV_NAME = "lbforaging:Foraging-8x8-3p-2f-v3"
TOTAL_STEPS = 1_000_000
ALGORITHM = "ia2c"
N_JOBS = 3   # number of parallel processes
BASE_OUTDIR = Path("outputs/multiruns_joblib")
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------
def make_tag(lr, entropy):
    """Format run directory label."""
    lr_tag = f"a{str(lr).replace('0.', '')[-4:]}"
    e_tag = f"e{str(entropy).replace('0.', '')[-3:]}"
    return f"{lr_tag}{e_tag}"


def run_experiment(lr, entropy, gamma, seed):
    """Run a single experiment as a subprocess."""
    tag = make_tag(lr, entropy)
    tag+="_l256_long"
    outdir = BASE_OUTDIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, RUN_SCRIPT,
        f"+algorithm={ALGORITHM}",
        f"env.name={ENV_NAME}",
        "env.time_limit=50",
        "env.standardise_rewards=True",
        f"algorithm.total_steps={TOTAL_STEPS}",
        f"algorithm.name={ALGORITHM}",
        "algorithm.video_interval=200000",
        f"algorithm.lr={lr}",
        f"algorithm.entropy_coef={entropy}",
        f"algorithm.gamma={gamma}",
        f"seed={seed}",
        f"hydra.run.dir={outdir.as_posix()}",
    ]

    print(f"\n[Joblib] Launching {tag} → {outdir}")
    # Run process with stdout piped directly to console (realtime logs)
    result = subprocess.run(cmd, cwd=os.getcwd())
    print(f"[Joblib] Finished {tag} → exit code {result.returncode}")
    return result.returncode


def main():
    combos = list(product(LRS, ENTROPY_COEFS, GAMMAS))
    print(f"Running {len(combos)} experiments ({N_JOBS} in parallel)")

    Parallel(n_jobs=N_JOBS)(
        delayed(run_experiment)(lr, entropy, gamma, SEED)
        for (lr, entropy, gamma) in combos
    )

    print("\nAll sweeps complete")


if __name__ == "__main__":
    main()
