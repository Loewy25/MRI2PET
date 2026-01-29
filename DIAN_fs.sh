#!/bin/bash
#SBATCH --job-name=DIAN_FS
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=2%10
#SBATCH --output=slurm-%A_%a_fs.out
#SBATCH --error=slurm-%A_%a_fs.out

set -e

# ---------------- FreeSurfer (SIMPLE, LIKE THE WORKING SCRIPT) ----------------
module purge
module load freesurfer/7.4.1
export FS_LICENSE=/ceph/chpc/mapped/brier/software/freesurfer/license.txt

# ---------------- Paths ----------------
LIST="/scratch/l.peiwang/dian_geom_all_t1_list.tsv"
SUBJECTS_DIR="/scratch/l.peiwang/freesurfer_DIAN"
EXPORT_ROOT="/scratch/l.peiwang/DIAN_fs_exports"

mkdir -p "$SUBJECTS_DIR" "$EXPORT_ROOT"
export SUBJECTS_DIR

# ---------------- Get this array row ----------------
line=$(awk -v n="$SLURM_ARRAY_TASK_ID" 'NR==n+1{print}' "$LIST")
[ -z "$line" ] && exit 0

sess_id=$(echo "$line" | cut -f1)
series=$(echo "$line" | cut -f2)
nii=$(echo "$line" | cut -f4)

FSID="${sess_id}__${series}"

echo "FSID=$FSID"
echo "T1=$nii"
echo "Start: $(date)"

# ---------------- Run FreeSurfer ----------------
recon-all \
  -sd "$SUBJECTS_DIR" \
  -s "$FSID" \
  -i "$nii" \
  -3T \
  -all \
  -threads "$SLURM_CPUS_PER_TASK"

# ---------------- Export (minimal) ----------------
OUT="$EXPORT_ROOT/$FSID"
mkdir -p "$OUT"

mri_convert "$SUBJECTS_DIR/$FSID/mri/orig.mgz"       "$OUT/T1_fs_orig.nii.gz"
mri_convert "$SUBJECTS_DIR/$FSID/mri/brainmask.mgz"  "$OUT/brainmask.nii.gz"
mri_convert "$SUBJECTS_DIR/$FSID/mri/aseg.mgz"       "$OUT/aseg.nii.gz"
mri_convert "$SUBJECTS_DIR/$FSID/mri/aparc+aseg.mgz" "$OUT/aparc_aseg.nii.gz"

echo "Finished: $(date)"

