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

# Always keep core Unix tools available (FSL wrapper scripts need basename/tr/etc.)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# Load FSL first
module load fsl

# Load FreeSurfer module (even if FREESURFER_HOME is misleading on compute nodes)
module load freesurfer || true

# ---- Force-add likely FreeSurfer binary locations (compute-node reality) ----
# Put these FIRST so `which mri_vol2vol` finds them if they exist.
CAND_FS_BIN_DIRS=(
  "/export/freesurfer/8.1.0"
  "/export/freesurfer/freesurfer-7.4.1/bin"
  "/export/freesurfer/freesurfer-7.2.0/bin"
  "/export/freesurfer/freesurfer-7.1.1/bin"
  "/export/freesurfer/freesurfer-6.0.0/bin"
  "/export/freesurfer/freesurfer-5.3.0/bin"
  "/export/freesurfer/freesurfer-5.3.0-HCP/bin"
)

for d in "${CAND_FS_BIN_DIRS[@]}"; do
  if [ -d "$d" ]; then
    export PATH="$d:$PATH"
  fi
done
hash -r

# ---- Pick a license file that exists ON THIS NODE ----
CAND_LICENSES=(
  "/export/freesurfer/8.1.0/.license"
  "/export/freesurfer/8.1.0/license.txt"
  "/export/freesurfer/freesurfer-7.4.1/.license"
  "/export/freesurfer/freesurfer-7.2.0/.license"
  "/export/freesurfer/freesurfer-7.1.1/.license"
  "/export/freesurfer/freesurfer-6.0.0/.license"
  "/export/freesurfer/freesurfer-5.3.0/.license"
  "/export/freesurfer/freesurfer-5.3.0-HCP/.license"
)

FS_LICENSE_FOUND=""
for p in "${CAND_LICENSES[@]}"; do
  if [ -f "$p" ]; then
    FS_LICENSE_FOUND="$p"
    break
  fi
done

if [ -z "$FS_LICENSE_FOUND" ]; then
  echo "ERROR: No FreeSurfer license file found on this node."
  echo "Tried:"
  printf "  %s\n" "${CAND_LICENSES[@]}"
  echo "Debug: listing /export/freesurfer (if accessible)"
  ls -la /export/freesurfer 2>&1 || true
  exit 3
fi
export FS_LICENSE="$FS_LICENSE_FOUND"

# ---- Conda last ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

# ---- Preflight ----
echo "=== BASIC ENV ==="
echo "HOST=$(hostname)"
echo "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}"
echo "FS_LICENSE=$FS_LICENSE"
echo "PATH=$PATH"

echo "=== coreutils check ==="
for x in basename tr ls; do
  printf "%-10s" "$x"; command -v "$x" 2>&1 || echo "MISSING"
done

echo "=== tool check ==="
for x in flirt mri_vol2vol mri_robust_register; do
  printf "%-18s" "$x"; command -v "$x" 2>&1 || echo "MISSING"
done

# Hard fail if FS tools still missing
command -v mri_vol2vol >/dev/null 2>&1 || { echo "ERROR: mri_vol2vol still not found after PATH fixes."; exit 4; }
command -v mri_robust_register >/dev/null 2>&1 || { echo "ERROR: mri_robust_register still not found after PATH fixes."; exit 5; }

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
for x in ["flirt","mri_vol2vol","mri_robust_register"]:
    print(f"{x:18s} -> {shutil.which(x)}")
PY

# ---- Run ----
python Data_TAU_ALL.py
