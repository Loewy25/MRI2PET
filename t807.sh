#!/bin/bash
#SBATCH --job-name=KARI_TAU_ALL
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --array=1-10
#SBATCH --output=dataset_%A_%a.out
#SBATCH --error=dataset_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load fsl
module load freesurfer || true

# Base PATH
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# FreeSurfer license
export FS_LICENSE="$HOME/freesurfer_license.txt"
if [ ! -f "$FS_LICENSE" ]; then
  echo "ERROR: FS_LICENSE not found at $FS_LICENSE"
  echo "Fix: on login node run: cp /export/freesurfer/freesurfer-6.0.0/.license \$HOME/freesurfer_license.txt"
  exit 2
fi

# Add FreeSurfer installs if present
for d in /export/freesurfer/8.1.0 /export/freesurfer/7.3.2 /export/freesurfer/7.2.0 /export/freesurfer/7.1.1 /export/freesurfer/6.0.0 /export/freesurfer/5.3.0 /export/freesurfer/5.3.0-HCP; do
  if [ -d "$d" ]; then
    if [ -z "${FREESURFER_HOME:-}" ]; then
      export FREESURFER_HOME="$d"
    fi
    if [ -d "$d/bin" ]; then
      export PATH="$d/bin:$PATH"
    fi
    export PATH="$d:$PATH"
  fi
done
hash -r

# Conda last
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

echo "=== ENV ==="
echo "HOST=$(hostname)"
echo "FS_LICENSE=$FS_LICENSE"
echo "FREESURFER_HOME=${FREESURFER_HOME:-}"
echo "PATH=$PATH"

echo "=== tools ==="
for x in basename tr flirt mri_vol2vol mri_robust_register python; do
  printf "%-18s" "$x"; command -v "$x" || echo "MISSING"
done

command -v flirt >/dev/null 2>&1 || { echo "ERROR: flirt not found on this node."; exit 3; }
command -v mri_vol2vol >/dev/null 2>&1 || { echo "ERROR: mri_vol2vol not found on this node."; exit 4; }
command -v mri_robust_register >/dev/null 2>&1 || { echo "ERROR: mri_robust_register not found on this node."; exit 5; }

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
for x in ["python", "flirt", "mri_vol2vol", "mri_robust_register"]:
    print(f"{x:18s} -> {shutil.which(x)}")
PY

NUM_PARTS="${NUM_PARTS:-10}"
PART="${PART:-${SLURM_ARRAY_TASK_ID:-1}}"

echo "Running chunk ${PART}/${NUM_PARTS}"

python DATA_TAU_ALL.py --part "$PART" --num-parts "$NUM_PARTS"
