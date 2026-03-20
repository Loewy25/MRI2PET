#!/bin/bash -l
#SBATCH --job-name=DIAN_PET
#SBATCH --mem=16G
#SBATCH --time=4:50:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --exclude=node19
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a_dian_pet.out
#SBATCH --error=slurm-%A_%a_dian_pet.out

set -euo pipefail

require_bin() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  echo "[ERROR] missing required binary: $name" >&2
  return 1
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"
PY_SCRIPT="$REPO_DIR/dian_pet.py"

echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-NA}"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_DIR=$REPO_DIR"
echo "PY_SCRIPT=$PY_SCRIPT"

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile ] && source /etc/profile
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi

module purge
module load fsl

export FSLOUTPUTTYPE=NIFTI_GZ

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

echo "=== Sanity: binaries ==="
require_bin python
require_bin mcflirt
require_bin dcm2niix

echo "=== Sanity: data mounts ==="
ls -ld /ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids
ls -ld /scratch/l.peiwang/DIAN_spreadsheet
mkdir -p /scratch/l.peiwang/DIAN_PET
ls -ld /scratch/l.peiwang/DIAN_PET

echo "=== Sanity: code ==="
ls -l "$PY_SCRIPT"

echo "=== module list ==="
module list

EXTRA_ARGS=${DIAN_PET_ARGS:-}

# shellcheck disable=SC2086
python -u "$PY_SCRIPT" \
  --num-tasks "${SLURM_ARRAY_TASK_COUNT:-10}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  $EXTRA_ARGS
