#!/bin/bash
#SBATCH --job-name=MGDA_UB
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --output=dataset.out
#SBATCH --error=dataset.err

module purge

# Load tool modules first
module load fsl
module load freesurfer

# Probably unnecessary on a CPU partition unless you truly need them
# module load cuda/12.9
# module load cudnn/9.11.0.98-cuda12

# Re-apply FreeSurfer environment after all module loads
if [ -n "$FREESURFER_HOME" ] && [ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]; then
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
fi

# Activate conda LAST so python/pip come from your env, not FSL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta

export FSLOUTPUTTYPE=NIFTI_GZ
export PYTHONNOUSERSITE=1

echo "=== final tool check ==="
for x in python pip flirt mri_vol2vol mri_robust_register; do
  printf "%-22s" "$x"
  which "$x" || echo "MISSING"
done

python - <<'PY'
import os, sys, shutil
print("sys.executable =", sys.executable)
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
for x in ["flirt", "mri_vol2vol", "mri_robust_register"]:
    print(f"{x} -> {shutil.which(x)}")
PY

# Do NOT pip install torch inside the job
python DATA_TAU_ALL.py
