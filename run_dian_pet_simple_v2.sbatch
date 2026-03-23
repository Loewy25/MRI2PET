#!/bin/bash -l
#SBATCH --job-name=DIAN_PET
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --exclude=node19
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a_dian_pet.out
#SBATCH --error=slurm-%A_%a_dian_pet.out

set -euo pipefail


module purge
module load fsl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

python -u dian_pet_simple_v2.py \
  --root /ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids \
  --out-root /scratch/l.peiwang/DIAN_PET \
  --num-tasks "${SLURM_ARRAY_TASK_COUNT:-10}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --max-4d-frames 8
