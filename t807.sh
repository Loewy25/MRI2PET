#!/bin/bash
#SBATCH --job-name=KARI_TAU_ALL
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --output=dataset.out
#SBATCH --error=dataset.err

set -u  # avoid -e so debug can print even if something fails

module purge
module load fsl
module load freesurfer

# Core Unix tools for FSL wrapper scripts
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# FreeSurfer license (cluster-provided)
export FS_LICENSE="/export/freesurfer/freesurfer-7.4.1/.license"

echo "=== BASIC ENV ==="
echo "HOST=$(hostname)"
echo "FREESURFER_HOME=${FREESURFER_HOME:-<unset>}"
echo "FS_LICENSE=$FS_LICENSE"
echo "PATH=$PATH"

# Debug FreeSurfer home visibility (DO NOT FAIL the job on errors)
echo "=== CHECK FREESURFER_HOME ACCESS ==="
ls -la "${FREESURFER_HOME:-/nope}" 2>&1 || true
ls -la "${FREESURFER_HOME:-/nope}/bin" 2>&1 || true

# If bin exists, force it to the front (prevents stray 8.1.0)
if [ -n "${FREESURFER_HOME:-}" ] && [ -d "$FREESURFER_HOME/bin" ]; then
  export PATH="$FREESURFER_HOME/bin:$PATH"
  hash -r
fi

# Fail fast ONLY on license file (this must exist)
if [ ! -f "$FS_LICENSE" ]; then
  echo "ERROR: FreeSurfer license missing at $FS_LICENSE"
  exit 2
fi

# Conda last
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

echo "=== TOOL CHECK ==="
for x in basename tr ls python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-20s" "$x"; command -v "$x" 2>&1 || echo "MISSING"
done

echo "=== which -a FreeSurfer tools ==="
which -a mri_vol2vol 2>&1 || true
which -a mri_robust_register 2>&1 || true

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("FS_LICENSE =", os.environ.get("FS_LICENSE"))
for x in ["flirt","mri_vol2vol","mri_robust_register"]:
    print(f"{x:18s} -> {shutil.which(x)}")
PY

# Now run (turn on -e only for the actual workload)
set -e
python Data_TAU_ALL.py
