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

# Ensure core Unix tools exist (FSL wrapper scripts need basename/tr/etc.)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# ---- Discover FreeSurfer from the binary that EXISTS on the compute node ----
# (Do NOT trust FREESURFER_HOME from module if the FS tree isn't mounted.)
FS_VOL2VOL="$(command -v mri_vol2vol || true)"
FS_RR="$(command -v mri_robust_register || true)"

echo "=== BASIC ENV ==="
echo "HOST=$(hostname)"
echo "PATH=$PATH"
echo "mri_vol2vol=$FS_VOL2VOL"
echo "mri_robust_register=$FS_RR"
echo "flirt=$(command -v flirt || true)"

if [ -z "$FS_VOL2VOL" ]; then
  echo "ERROR: mri_vol2vol not found on PATH on this node."
  exit 2
fi

FS_BIN_DIR="$(dirname "$FS_VOL2VOL")"
# Common layouts:
#  /export/freesurfer/8.1.0/mri_vol2vol  -> FS_ROOT=/export/freesurfer/8.1.0
#  /something/.../bin/mri_vol2vol        -> FS_ROOT=/something/...
FS_ROOT="$(cd "$FS_BIN_DIR" && pwd)"

echo "FS_BIN_DIR=$FS_BIN_DIR"
echo "FS_ROOT(guessed)=$FS_ROOT"

# ---- Find a license file that exists on THIS node ----
# Check a few common locations relative to the binary location.
CAND_LICENSES=(
  "$FS_ROOT/.license"
  "$FS_ROOT/license.txt"
  "$FS_ROOT/../.license"
  "$FS_ROOT/../license.txt"
  "$FS_ROOT/../../.license"
  "$FS_ROOT/../../license.txt"
)

FS_LICENSE_FOUND=""
for p in "${CAND_LICENSES[@]}"; do
  if [ -f "$p" ]; then
    FS_LICENSE_FOUND="$p"
    break
  fi
done

if [ -z "$FS_LICENSE_FOUND" ]; then
  echo "ERROR: Could not find a FreeSurfer license near $FS_ROOT"
  echo "Tried:"
  printf '  %s\n' "${CAND_LICENSES[@]}"
  echo "Directory listings:"
  ls -la "$FS_ROOT" 2>&1 || true
  ls -la "$(dirname "$FS_ROOT")" 2>&1 || true
  exit 3
fi

export FS_LICENSE="$FS_LICENSE_FOUND"
echo "FS_LICENSE=$FS_LICENSE"

# Optional: set FREESURFER_HOME for tools that care (not strictly required)
export FREESURFER_HOME="$FS_ROOT"
echo "FREESURFER_HOME=$FREESURFER_HOME"

# ---- Conda last ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

echo "=== TOOL CHECK ==="
for x in basename tr ls python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-20s" "$x"
  command -v "$x" 2>&1 || echo "MISSING"
done

echo "=== RUN ==="
python Data_TAU_ALL.py
