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

# --- Modules ---
module load fsl
module load freesurfer

# --- Make sure core Unix tools exist (basename/tr/etc) ---
# (FSL wrapper scripts call these; some environments end up with a stripped PATH)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# --- FreeSurfer setup + license (cluster-provided license exists) ---
# Module sets FREESURFER_HOME; always source SetUpFreeSurfer.sh
echo "FREESURFER_HOME(from module)=$FREESURFER_HOME"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

export FS_LICENSE="/export/freesurfer/freesurfer-7.4.1/.license"
test -f "$FS_LICENSE" || { echo "ERROR: FS_LICENSE missing: $FS_LICENSE"; exit 1; }

# --- Force FreeSurfer 7.4.1 binaries to the FRONT (avoid accidental 8.1.0 on PATH) ---
# After sourcing SetUpFreeSurfer.sh, FS usually defines $FREESURFER_HOME/bin; make it first anyway.
if [ -d "$FREESURFER_HOME/bin" ]; then
  export PATH="$FREESURFER_HOME/bin:$PATH"
fi
hash -r

# --- Conda last ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

# Avoid accidental ~/.local pollution
export PYTHONNOUSERSITE=1

# FSL output type used by FLIRT
export FSLOUTPUTTYPE=NIFTI_GZ

# --- Preflight checks ---
echo "=== coreutils check ==="
for x in basename tr awk sed; do
  printf "%-10s" "$x"
  which "$x" || echo "MISSING"
done

echo "=== tool check ==="
for x in python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-20s" "$x"
  which "$x" || echo "MISSING"
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

# --- Run ---
python Data_TAU_ALL.py
