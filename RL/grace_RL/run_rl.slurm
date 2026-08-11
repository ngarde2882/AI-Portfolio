#!/usr/bin/env bash
#SBATCH --job-name=rl_gauntlet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngarde2882@tamu.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.1.0
source activate rl_env

# ── Run directory (keep outputs on scratch for I/O performance) ───────────────
RUN_DIR="$SCRATCH/rl_runs/run_$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# Copy source files to scratch
cp -r "$SLURM_SUBMIT_DIR"/*.py "$RUN_DIR"/

# Copy existing checkpoints if available (resume training)
if [ -d "$SLURM_SUBMIT_DIR/agents" ]; then
    cp -r "$SLURM_SUBMIT_DIR/agents" "$RUN_DIR"/
fi
if [ -d "$SLURM_SUBMIT_DIR/ac_agents" ]; then
    cp -r "$SLURM_SUBMIT_DIR/ac_agents" "$RUN_DIR"/
fi

cd "$RUN_DIR"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Run dir: $RUN_DIR"

# ── Training ──────────────────────────────────────────────────────────────────
python RL.py \
    --games_per_matchup 5 \
    --gauntlet_repeats  20 \
    --training_cycles   5 \
    --out_dir   "$RUN_DIR/out" \
    --agent_dir "$RUN_DIR/agents" \
    --device    cuda \
    --seed      42

echo "Training complete at $(date)"

# ── Copy outputs back to submit dir ───────────────────────────────────────────
cp -r "$RUN_DIR/out"       "$SLURM_SUBMIT_DIR/out_$SLURM_JOB_ID"
cp -r "$RUN_DIR/agents"    "$SLURM_SUBMIT_DIR/agents_$SLURM_JOB_ID"
cp -r "$RUN_DIR/ac_agents" "$SLURM_SUBMIT_DIR/ac_agents_$SLURM_JOB_ID" 2>/dev/null || true

echo "Outputs copied back to $SLURM_SUBMIT_DIR"
