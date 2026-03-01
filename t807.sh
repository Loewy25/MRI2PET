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
module load freesurfer

# Ensure core Unix tools exist (FSL wrappers need basename/tr/etc.)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# ---------------- FreeSurfer setup ----------------
echo "FREESURFER_HOME(from module)=${FREESURFER_HOME:-<unset>}"

# Find a usable SetUpFreeSurfer script (cluster layouts vary)
FS_SETUP=""
for cand in \
  "$FREESURFER_HOME/SetUpFreeSurfer.sh" \
  "$FREESURFER_HOME/SetUpFreeSurfer.bash" \
  "$FREESURFER_HOME/SetUpFreeSurfer" \
  "$FREESURFER_HOME/bin/SetUpFreeSurfer.sh" \
  "$FREESURFER_HOME/bin/SetUpFreeSurfer.bash" \
  "$FREESURFER_HOME/bin/SetUpFreeSurfer"
do
  if [ -f "$cand" ]; then
    FS_SETUP="$cand"
    break
  fi
done

if [ -z "$FS_SETUP" ]; then
  echo "ERROR: Could not find SetUpFreeSurfer.sh/.bash under $FREESURFER_HOME"
  echo "Top-level listing:"
  ls -la "$FREESURFER_HOME" || true
  echo "Bin listing:"
  ls -la "$FREESURFER_HOME/bin" || true
  exit 2
fi

echo "Using FreeSurfer setup: $FS_SETUP"
# shellcheck disable=SC1090
source "$FS_SETUP"

# Use the cluster-provided license you found
export FS_LICENSE="/export/freesurfer/freesurfer-7.4.1/.license"
test -f "$FS_LICENSE" || { echo "ERROR: FS_LICENSE missing: $FS_LICENSE"; exit 3; }

# Force THIS FreeSurfer version's bin first (avoid accidental /export/freesurfer/8.1.0)
if [ -d "$FREESURFER_HOME/bin" ]; then
  export PATH="$FREESURFER_HOME/bin:$PATH"
fi
hash -r

# ---------------- Conda last ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

# ---------------- Preflight ----------------
echo "=== coreutils check ==="
for x in basename tr awk sed; do
  printf "%-10s" "$x"; which "$x" || echo "MISSING"
done

echo "=== tool check ==="
for x in python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-20s" "$x"; which "$x" || echo "MISSING"
done

echo "=== which -a (FS tools) ==="
which -a mri_vol2vol || true
which -a mri_robust_register || true

echo "=== env summary ==="
echo "PATH=$PATH"
echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "FS_LICENSE=$FS_LICENSE"
echo "FSLOUTPUTTYPE=$FSLOUTPUTTYPE"

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
for x in ["basename","tr","flirt","mri_vol2vol","mri_robust_register"]:
    print(f"{x:18s} -> {shutil.which(x)}")
PY

# ---------------- Run ----------------
python Data_TAU_ALL.py
