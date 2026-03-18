#!/bin/bash -l
#SBATCH --job-name=DIAN_ALL
#SBATCH --mem=20G
#SBATCH --time=23:50:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a_dian_all.out
#SBATCH --error=slurm-%A_%a_dian_all.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID:-NA}"
echo "TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-NA}"
echo "PWD=$(pwd)"

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile ] && source /etc/profile
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi

module purge
module load freesurfer
module load fsl

echo "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}"
if [ -n "${FREESURFER_HOME:-}" ] && [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
  # shellcheck disable=SC1090
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
fi
if [ -n "${FREESURFER_HOME:-}" ]; then
  export PATH="${FREESURFER_HOME}/bin:${PATH}"
fi

export FSLOUTPUTTYPE=NIFTI_GZ

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

echo "=== Sanity: binaries ==="
command -v python
command -v flirt
command -v mri_vol2vol
command -v mri_robust_register

echo "=== Sanity: data mounts ==="
ls -ld /scratch/l.peiwang/DIAN_PET
ls -ld /scratch/l.peiwang/DIAN_fs
ls -ld /scratch/l.peiwang/DIAN_spreadsheet
ls -ld /scratch/l.peiwang/DIAN_ALL_FINISHED || true

echo "=== module list ==="
module list

python -u Data_DIAN_ALL.py \
  --num-tasks "${SLURM_ARRAY_TASK_COUNT:-10}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}"
