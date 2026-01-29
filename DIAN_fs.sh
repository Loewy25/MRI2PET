#!/bin/bash
#SBATCH --job-name=DIAN_FS
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-2983%20
#SBATCH --output=slurm-%A_%a_fs.out
#SBATCH --error=slurm-%A_%a_fs.out

set -euo pipefail

LIST="/scratch/l.peiwang/dian_geom_all_t1_list.tsv"
FS_SUBJECTS_DIR="/scratch/l.peiwang/freesurfer_DIAN"
EXPORT_ROOT="/scratch/l.peiwang/DIAN_fs_exports"

mkdir -p "$FS_SUBJECTS_DIR" "$EXPORT_ROOT"

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "[INFO] LIST=$LIST"
echo "[INFO] SUBJECTS_DIR=$FS_SUBJECTS_DIR"
echo "[INFO] EXPORT_ROOT=$EXPORT_ROOT"

# --- module init (safe with set -u) ---
set +u
for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
  [[ -r "$f" ]] && source "$f" || true
done
set -u

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load freesurfer >/dev/null 2>&1 || true
  module list 2>/dev/null || true
fi

if [[ -n "${FREESURFER_HOME:-}" && -r "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]]; then
  # shellcheck disable=SC1091
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
fi

export SUBJECTS_DIR="$FS_SUBJECTS_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

command -v recon-all >/dev/null 2>&1 || { echo "[FATAL] recon-all not found"; exit 2; }
command -v mri_convert >/dev/null 2>&1 || { echo "[FATAL] mri_convert not found"; exit 2; }

# ---- read row N (skip header) ----
line="$(awk -v n="${SLURM_ARRAY_TASK_ID}" 'NR==n+1{print; exit}' "$LIST" || true)"
[[ -n "$line" ]] || { echo "[DONE] no row for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

sess_id="$(cut -f1 <<<"$line")"
series="$(cut -f2 <<<"$line")"
nii_link="$(cut -f3 <<<"$line")"
nii="$(cut -f4 <<<"$line")"   # <-- resolved real file target

subj="${sess_id}__${series}"
subj_safe="$(echo "$subj" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"

echo "[INFO] sess_id=$sess_id series=$series"
echo "[INFO] nii_link=$nii_link"
echo "[INFO] nii_target=$nii"
echo "[INFO] subject=$subj_safe"

[[ -f "$nii" ]] || { echo "[FAIL] target missing: $nii (link=$nii_link)"; exit 3; }

# ---- recon-all ----
if [[ -d "$SUBJECTS_DIR/$subj_safe" && -f "$SUBJECTS_DIR/$subj_safe/scripts/recon-all.done" ]]; then
  echo "[SKIP] recon-all already done for $subj_safe"
else
  recon-all -s "$subj_safe" -i "$nii" -all
fi

# ---- export key outputs ----
exp_dir="$EXPORT_ROOT/$subj_safe"
mkdir -p "$exp_dir"

mri_convert "$SUBJECTS_DIR/$subj_safe/mri/orig.mgz"       "$exp_dir/T1_fs_orig.nii.gz"
mri_convert "$SUBJECTS_DIR/$subj_safe/mri/brainmask.mgz"  "$exp_dir/brainmask.nii.gz"
mri_convert "$SUBJECTS_DIR/$subj_safe/mri/aseg.mgz"       "$exp_dir/aseg.nii.gz"
mri_convert "$SUBJECTS_DIR/$subj_safe/mri/aparc+aseg.mgz" "$exp_dir/aparc_aseg.nii.gz"

echo "[OK] exported -> $exp_dir"
