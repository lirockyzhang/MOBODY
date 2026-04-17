#!/bin/bash
#SBATCH --job-name=mobody-fg
#SBATCH -p gpu-common                    # or: scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/work/lz256/MOBODY/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/lz256/MOBODY/slurm_logs/%x-%A_%a.err

set -euo pipefail

# Project root on DCC
cd /work/lz256/MOBODY
mkdir -p slurm_logs runs

# Conda env (matches README install: `conda create -n offdynamics ...`)
module load Anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate offdynamics

# ---- sweep definition -------------------------------------------------------
# 1 policy x 2 envs x 2 shift_types x 2 shift_levels x 1 seed = 8 jobs
POLICIES=(MOBODY)
ENVS=(hopper halfcheetah)
SHIFTS=(friction gravity)
LEVELS=(0.5 2.0)
SEED=1

N_LEVELS=${#LEVELS[@]}
N_SHIFTS=${#SHIFTS[@]}
N_ENVS=${#ENVS[@]}
N_POLICIES=${#POLICIES[@]}

# decode SLURM_ARRAY_TASK_ID into (policy, env, shift, level)
idx=${SLURM_ARRAY_TASK_ID}
lvl_i=$(( idx % N_LEVELS )); idx=$(( idx / N_LEVELS ))
sh_i=$(( idx % N_SHIFTS )); idx=$(( idx / N_SHIFTS ))
env_i=$(( idx % N_ENVS ));   idx=$(( idx / N_ENVS ))
pol_i=$(( idx % N_POLICIES ))

POLICY=${POLICIES[$pol_i]}
ENV_BASE=${ENVS[$env_i]}
SHIFT=${SHIFTS[$sh_i]}
LEVEL=${LEVELS[$lvl_i]}
ENV_NAME="${ENV_BASE}-${SHIFT}"

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} policy=${POLICY} env=${ENV_NAME} shift_level=${LEVEL} seed=${SEED}"
nvidia-smi || true

# ---- policy-specific flags (from README) -----------------------------------
COMMON_ARGS=(
  --policy "${POLICY}"
  --env   "${ENV_NAME}"
  --shift_level "${LEVEL}"
  --seed  "${SEED}"
  --dir   runs
  --wandb 0
)

if [[ "${POLICY}" == "MOBODY" ]]; then
  EXTRA_ARGS=(
    --train_dynamics 1
    --penalty_type dara
    --env_penalty_coef 5
    --src_rollout_length 1
    --trg_rollout_length 1
    --bc_coef 1
  )
else  # DARA
  EXTRA_ARGS=(
    --train_dynamics 0
    --penalty_type dara
  )
fi

srun python -u train_mobody.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
