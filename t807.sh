#!/bin/bash
#SBATCH --job-name=KARI_TAU_ALL
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --output=dataset.out
#SBATCH --error=dataset.err

set -euo pipefail

module purge
module load fsl
module load freesurfer || true

# FSL wrapper needs these
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# ---- FreeSurfer license: use a file in HOME (compute nodes can see it) ----
export FS_LICENSE="$HOME/freesurfer_license.txt"
if [ ! -f "$FS_LICENSE" ]; then
  echo "ERROR: FS_LICENSE not found at $FS_LICENSE"
  echo "Fix: on login node run: cp /export/freesurfer/freesurfer-6.0.0/.license \$HOME/freesurfer_license.txt"
  exit 2
fi

# ---- Ensure FreeSurfer binaries are reachable ----
# (Your cluster sometimes has wrappers in /export/freesurfer/<ver> folders)
# Add the mounted versions’ dirs if they exist on compute node:
for d in /export/freesurfer/8.1.0 /export/freesurfer/7.3.2 /export/freesurfer/7.2.0 /export/freesurfer/7.1.1 /export/freesurfer/6.0.0 /export/freesurfer/5.3.0 /export/freesurfer/5.3.0-HCP; do
  if [ -d "$d" ]; then
    export PATH="$d:$PATH"
  fi
done
hash -r

# ---- Conda last ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

# ---- Preflight ----
echo "=== ENV ==="
echo "HOST=$(hostname)"
echo "FS_LICENSE=$FS_LICENSE"
echo "PATH=$PATH"

echo "=== tools ==="
for x in basename tr flirt mri_vol2vol mri_robust_register; do
  printf "%-18s" "$x"; command -v "$x" || echo "MISSING"
done

# Hard fail if FS tools missing
command -v mri_vol2vol >/dev/null 2>&1 || { echo "ERROR: mri_vol2vol not found on this node. FreeSurfer not available on this partition/node."; exit 3; }
command -v mri_robust_register >/dev/null 2>&1 || { echo "ERROR: mri_robust_register not found on this node."; exit 4; }

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
for x in ["flirt","mri_vol2vol","mri_robust_register"]:
    print(f"{x:18s} -> {shutil.which(x)}")
PY

python DATA_TAU_ALL.py
