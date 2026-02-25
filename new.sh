#!/bin/bash -l
#SBATCH --job-name=TAU_ALL
#SBATCH --time=23:50:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --account=shinjini_kundu
#SBATCH --partition=tier1
#SBATCH --array=1-4
#SBATCH --output=tau_%A_%a.out
#SBATCH --error=tau_%A_%a.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "PWD=$(pwd)"

# --- modules (login shell usually gives 'module', but keep this robust) ---
if ! command -v module >/dev/null 2>&1; then
  # Try the common init locations
  [ -f /etc/profile ] && source /etc/profile
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi

module purge
module load freesurfer
module load fsl

# --- FreeSurfer setup ---
echo "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}"
if [ -n "${FREESURFER_HOME:-}" ] && [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
fi
# Belt + suspenders: ensure FS binaries are in PATH
if [ -n "${FREESURFER_HOME:-}" ]; then
  export PATH="${FREESURFER_HOME}/bin:${PATH}"
fi

export FSLOUTPUTTYPE=NIFTI_GZ

# --- Conda env ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

# --- Required filesystem sanity checks (fail early) ---
mkdir -p /scratch/l.peiwang/kari_brainv33

echo "=== Sanity checks ==="
command -v mri_vol2vol
command -v mri_robust_register
command -v flirt

ls -ld /ceph/chpc/mapped/benz04_kari || true
ls -ld /ceph/chpc/mapped/benz04_kari/pup
ls -ld /ceph/chpc/mapped/benz04_kari/freesurfers

echo "=== module list ==="
module list

echo "=== Running pipeline part ${SLURM_ARRAY_TASK_ID} ==="
python -u Data_TAU_ALL.py --part "${SLURM_ARRAY_TASK_ID}"
