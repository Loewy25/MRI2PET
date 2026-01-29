#!/bin/bash
#SBATCH --job-name=DIAN_FS
#SBATCH --partition=tier1_cpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-1%1
#SBATCH --output=slurm-%A_%a_fs.out
#SBATCH --error=slurm-%A_%a_fs.out

set -euo pipefail

module load apptainer

# IMPORTANT: use the copy on /scratch (compute nodes may not see /export)
SIF="/scratch/l.peiwang/containers/freesurfer-7.2.0.linux-amd64.sif"

LIST="/scratch/l.peiwang/dian_geom_all_t1_list.tsv"
FS_SUBJECTS_DIR="/scratch/l.peiwang/freesurfer_DIAN"
EXPORT_ROOT="/scratch/l.peiwang/DIAN_fs_exports"
TMP_BASE="/scratch/l.peiwang/tmp_freesurfer"

mkdir -p "$FS_SUBJECTS_DIR" "$EXPORT_ROOT" "$TMP_BASE"

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "[INFO] SIF=$SIF"
echo "[INFO] LIST=$LIST"
echo "[INFO] SUBJECTS_DIR=$FS_SUBJECTS_DIR"
echo "[INFO] EXPORT_ROOT=$EXPORT_ROOT"

if [[ ! -r "$SIF" ]]; then
  echo "[FATAL] SIF not readable on this node: $SIF"
  echo "[HINT] On login node run:"
  echo "       mkdir -p /scratch/l.peiwang/containers"
  echo "       cp -av /export/freesurfer/freesurfer-7.2.0.linux-amd64.sif /scratch/l.peiwang/containers/"
  exit 2
fi

# ---- read row N (skip header) ----
line="$(awk -v n="${SLURM_ARRAY_TASK_ID}" 'NR==n+1{print; exit}' "$LIST" || true)"
[[ -n "$line" ]] || { echo "[DONE] no row for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

sess_id="$(cut -f1 <<<"$line")"
series="$(cut -f2 <<<"$line")"
nii_link="$(cut -f3 <<<"$line")"
nii="$(cut -f4 <<<"$line")"

subj="${sess_id}__${series}"
subj_safe="$(echo "$subj" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"

echo "[INFO] sess_id=$sess_id series=$series"
echo "[INFO] nii_link=$nii_link"
echo "[INFO] nii_target=$nii"
echo "[INFO] subject=$subj_safe"

if [[ ! -f "$nii" ]]; then
  echo "[FAIL] target missing: $nii (link=$nii_link)"
  exit 3
fi

tmp_dir="$TMP_BASE/$subj_safe"
mkdir -p "$tmp_dir"

# Bind /scratch so container can access your inputs/outputs/temp
BIND="/scratch:/scratch"

apptainer exec --bind "$BIND" "$SIF" bash -lc "
  set -euo pipefail
  export SUBJECTS_DIR='$FS_SUBJECTS_DIR'
  export OMP_NUM_THREADS='${SLURM_CPUS_PER_TASK:-1}'
  export TMPDIR='$tmp_dir'

  echo '[IN_CONTAINER] recon-all=' \$(command -v recon-all)
  echo '[IN_CONTAINER] mri_convert=' \$(command -v mri_convert)
  recon-all -version || true

  if [[ -d \"\$SUBJECTS_DIR/$subj_safe\" && -f \"\$SUBJECTS_DIR/$subj_safe/scripts/recon-all.done\" ]]; then
    echo '[SKIP] recon-all already done for $subj_safe'
  else
    recon-all -s '$subj_safe' -i '$nii' -all
  fi

  exp_dir='$EXPORT_ROOT/$subj_safe'
  mkdir -p \"\$exp_dir\"

  mri_convert \"\$SUBJECTS_DIR/$subj_safe/mri/orig.mgz\"       \"\$exp_dir/T1_fs_orig.nii.gz\"
  mri_convert \"\$SUBJECTS_DIR/$subj_safe/mri/brainmask.mgz\"  \"\$exp_dir/brainmask.nii.gz\"
  mri_convert \"\$SUBJECTS_DIR/$subj_safe/mri/aseg.mgz\"       \"\$exp_dir/aseg.nii.gz\"
  mri_convert \"\$SUBJECTS_DIR/$subj_safe/mri/aparc+aseg.mgz\" \"\$exp_dir/aparc_aseg.nii.gz\"

  echo '[OK] exported ->' \"\$exp_dir\"
"

echo "[INFO] done task ${SLURM_ARRAY_TASK_ID}"


