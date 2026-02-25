#!/bin/bash -l
#SBATCH --job-name=TAU_ALL
#SBATCH --mem=20G
#SBATCH --time=23:50:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --output=dataset.out
#SBATCH --error=dataset.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID}"
echo "PWD=$(pwd)"

# ---- Make 'module' available even in non-login batch contexts (belt+suspenders) ----
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile ] && source /etc/profile
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
  [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi

module purge

# ---- Load FreeSurfer + FSL (order matters: FS first) ----
module load freesurfer
module load fsl

# ---- Initialize FreeSurfer environment (ensures mri_vol2vol, mri_robust_register in PATH) ----
echo "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}"
if [ -n "${FREESURFER_HOME:-}" ] && [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then
  # shellcheck disable=SC1090
  source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
fi
# extra safety
if [ -n "${FREESURFER_HOME:-}" ]; then
  export PATH="${FREESURFER_HOME}/bin:${PATH}"
fi

export FSLOUTPUTTYPE=NIFTI_GZ

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

# ---- Sanity checks (fail early with useful logs) ----
echo "=== Sanity: binaries ==="
command -v mri_vol2vol
command -v mri_robust_register
command -v flirt

echo "=== Sanity: data mounts ==="
ls -ld /ceph/chpc/mapped/benz04_kari || true
ls -ld /ceph/chpc/mapped/benz04_kari/pup
ls -ld /ceph/chpc/mapped/benz04_kari/freesurfers

echo "=== module list ==="
module list

# ---- Run (single job, no arrays) ----
python -u Data_TAU_ALL.py
