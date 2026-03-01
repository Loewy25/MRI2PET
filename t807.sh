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

# --- Load tools (order matters less now, but keep it clean) ---
module load fsl
module load freesurfer

# --- FreeSurfer environment + license (use the cluster-provided .license) ---
# freesurfer module sets this:
echo "FREESURFER_HOME=$FREESURFER_HOME"

# Source FreeSurfer setup unconditionally
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# Point to the license that EXISTS on this cluster
export FS_LICENSE="/export/freesurfer/freesurfer-7.4.1/.license"

# --- Conda env (activate AFTER modules) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

# --- FSL output type for your FLIRT outputs ---
export FSLOUTPUTTYPE=NIFTI_GZ

# Avoid accidentally picking up ~/.local site-packages
export PYTHONNOUSERSITE=1

# --- Sanity checks (fail fast) ---
echo "=== FINAL TOOL CHECK ==="
for x in python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-20s" "$x"
  which "$x" || echo "MISSING"
done

echo "=== ENV CHECK ==="
echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "FS_LICENSE=$FS_LICENSE"
test -f "$FS_LICENSE" || { echo "ERROR: FS license missing: $FS_LICENSE"; exit 1; }

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
for x in ["flirt","mri_vol2vol","mri_robust_register"]:
    print(f"{x} -> {shutil.which(x)}")
PY

# --- Run your pipeline ---
python Data_TAU_ALL.py
