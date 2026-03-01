#!/bin/bash
#SBATCH --job-name=KARI_TAU_ALL
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --array=1-8
#SBATCH --output=dataset_%A_%a.out
#SBATCH --error=dataset_%A_%a.err

set -euo pipefail

module purge
module load fsl
module load freesurfer || true

# FSL wrappers need core utils (basename/tr/etc.)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
hash -r

# FreeSurfer license MUST be in HOME so compute nodes can read it
export FS_LICENSE="$HOME/freesurfer_license.txt"
if [ ! -f "$FS_LICENSE" ]; then
  echo "ERROR: FS_LICENSE not found at $FS_LICENSE"
  echo "Fix (run once on login node): cp /export/freesurfer/freesurfer-6.0.0/.license \$HOME/freesurfer_license.txt"
  exit 2
fi

# Add FreeSurfer wrapper dirs if present
for d in /export/freesurfer/8.1.0 /export/freesurfer/7.3.2 /export/freesurfer/7.2.0 /export/freesurfer/7.1.1 /export/freesurfer/6.0.0 /export/freesurfer/5.3.0 /export/freesurfer/5.3.0-HCP; do
  [ -d "$d" ] && export PATH="$d:$PATH"
done
hash -r

# Conda last
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pasta
export PYTHONNOUSERSITE=1
export FSLOUTPUTTYPE=NIFTI_GZ

echo "HOST=$(hostname)"
echo "PART=${SLURM_ARRAY_TASK_ID}/8"
echo "FS_LICENSE=$FS_LICENSE"
echo "flirt=$(command -v flirt || true)"
echo "mri_vol2vol=$(command -v mri_vol2vol || true)"
echo "mri_robust_register=$(command -v mri_robust_register || true)"

command -v mri_vol2vol >/dev/null 2>&1 || { echo "ERROR: mri_vol2vol not found"; exit 3; }
command -v mri_robust_register >/dev/null 2>&1 || { echo "ERROR: mri_robust_register not found"; exit 4; }

# CHANGE THIS LINE to the exact python filename you actually have:
python Data_TAU_ALL.py --nparts 8 --part "${SLURM_ARRAY_TASK_ID}"
