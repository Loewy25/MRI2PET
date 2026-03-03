#!/bin/bash
#SBATCH --job-name=fastsurfer_seg
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --output=slurm-%A_%a_fastsurfer.out
#SBATCH --error=slurm-%A_%a_fastsurfer.out

set -euo pipefail

# ============================================================
# Config
# ============================================================
IN_ROOT="${IN_ROOT:-/scratch/l.peiwang/DIAN_geom}"                   # contains top-level subject folders
OUT_ROOT="${OUT_ROOT:-/scratch/l.peiwang/fastsurfer_simple_out}"     # output root (will mirror IN_ROOT hierarchy)
SIF_IMAGE="${SIF_IMAGE:-/scratch/l.peiwang/fastsurfer-gpu.sif}"      # prebuilt .sif (must exist)
IMAGE_SRC="${IMAGE_SRC:-docker://deepmi/fastsurfer:latest}"          # only for manual one-time build outside array jobs
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SKIP_DONE="${SKIP_DONE:-1}"
ONLY_SUBJECT="${ONLY_SUBJECT:-}"                                     # optional: restrict to one top-level folder

# Constant sid INSIDE each mirrored folder (safe because --sd is unique per scan folder)
FS_SID="${FS_SID:-fastsurfer}"

# Split across 10 array tasks
N_SPLITS=10
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# FreeSurfer license on HOST (must exist). We bind it into container at /fs_license.txt
FS_LICENSE_HOST="${FS_LICENSE_HOST:-/ceph/chpc/mapped/brier/software/freesurfer/license.txt}"
FS_LICENSE_IN="/fs_license.txt"

# Bind broad roots so symlinks/data are visible in container
BIND_PATHS="${BIND_PATHS:-/scratch:/scratch}"
if [[ -d /ceph ]]; then
  BIND_PATHS="${BIND_PATHS},/ceph:/ceph"
fi

# ============================================================
# Helpers
# ============================================================
pick_first_existing() {
  local p
  for p in "$@"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
  return 1
}

convert_mgz_to_nii() {
  local src="$1"
  local dst="$2"
  "$CTR" exec --nv --no-home -e \
    -B "$BIND_PATHS" \
    -B "$FS_LICENSE_HOST:$FS_LICENSE_IN" \
    --env FS_LICENSE="$FS_LICENSE_IN" \
    "$SIF_IMAGE" \
    mri_convert "$src" "$dst"
}

# ============================================================
# Load Apptainer/Singularity
# ============================================================
set +u
for f in /etc/profile.d/modules.sh \
         /usr/share/Modules/init/bash \
         /usr/share/lmod/lmod/init/bash; do
  [[ -r "$f" ]] && source "$f" || true
done
set -u

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load apptainer/1.4.0 >/dev/null 2>&1 || true
  module load singularity/1.4.0 >/dev/null 2>&1 || true
fi

if command -v apptainer >/dev/null 2>&1; then
  CTR=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CTR=singularity
else
  echo "[FATAL] apptainer/singularity not found" >&2
  exit 1
fi

# ============================================================
# Preflight
# ============================================================
[[ -d "$IN_ROOT" ]] || { echo "[FATAL] IN_ROOT not found: $IN_ROOT" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

[[ -f "$FS_LICENSE_HOST" ]] || {
  echo "[FATAL] FS_LICENSE_HOST not found: $FS_LICENSE_HOST" >&2
  echo "        (Set FS_LICENSE_HOST to your license path, or copy one to \$HOME and point to it.)" >&2
  exit 2
}

# For an array job, require the image to already exist.
if [[ ! -f "$SIF_IMAGE" ]]; then
  echo "[FATAL] SIF missing: $SIF_IMAGE" >&2
  echo "        Build it once before submitting the 10-task array." >&2
  echo "        Example: $CTR build $SIF_IMAGE $IMAGE_SRC" >&2
  exit 2
fi

if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID >= N_SPLITS )); then
  echo "[FATAL] TASK_ID must be in 0..$((N_SPLITS - 1)); got: $TASK_ID" >&2
  exit 2
fi

# Avoid container warning about missing /home/...
cd /tmp

echo "[INFO] runtime:         $CTR"
echo "[INFO] IN_ROOT:         $IN_ROOT"
echo "[INFO] OUT_ROOT:        $OUT_ROOT"
echo "[INFO] SIF_IMAGE:       $SIF_IMAGE"
echo "[INFO] THREADS:         $THREADS"
echo "[INFO] BIND_PATHS:      $BIND_PATHS"
echo "[INFO] FS_LICENSE_HOST: $FS_LICENSE_HOST -> $FS_LICENSE_IN (in container)"
echo "[INFO] ONLY_SUBJECT:    ${ONLY_SUBJECT:-<all>}"
echo "[INFO] task:            $((TASK_ID + 1))/$N_SPLITS"
echo "[INFO] FS_SID:          $FS_SID"
echo "[INFO] start:           $(date)"

# ============================================================
# Enumerate top-level subject folders (immediate children)
# ============================================================
declare -a SUBJECT_DIRS=()
shopt -s nullglob
for d in "$IN_ROOT"/*; do
  [[ -d "$d" ]] || continue
  if [[ -n "$ONLY_SUBJECT" && "$(basename "$d")" != "$ONLY_SUBJECT" ]]; then
    continue
  fi
  SUBJECT_DIRS+=("$d")
done
shopt -u nullglob

if [[ ${#SUBJECT_DIRS[@]} -eq 0 ]]; then
  echo "[FATAL] no subject subfolders found under $IN_ROOT" >&2
  exit 3
fi

echo "[INFO] total matching subject folders: ${#SUBJECT_DIRS[@]}"

DONE=0
SKIPPED_DONE=0
SKIPPED_NO_T1=0
FAILED=0
ASSIGNED_SUBJECTS=0
DISCOVERED_T1=0

# ============================================================
# Main loop
# Split top-level subject folders across 10 tasks by index % 10
# ============================================================
for i in "${!SUBJECT_DIRS[@]}"; do
  (( i % N_SPLITS == TASK_ID )) || continue
  ASSIGNED_SUBJECTS=$((ASSIGNED_SUBJECTS + 1))

  SUB="${SUBJECT_DIRS[$i]}"

  echo ""
  echo "============================================================"
  echo "[INFO] subject folder: $SUB"
  echo "============================================================"

  # Process ALL T1.nii.gz files found anywhere under this subject folder
  mapfile -t T1S < <(find -L "$SUB" -type f -name 'T1.nii.gz' | sort)

  if [[ ${#T1S[@]} -eq 0 ]]; then
    echo "[WARN] no T1.nii.gz found inside $SUB"
    SKIPPED_NO_T1=$((SKIPPED_NO_T1 + 1))
    continue
  fi

  DISCOVERED_T1=$((DISCOVERED_T1 + ${#T1S[@]}))

  echo "[INFO] found ${#T1S[@]} T1.nii.gz file(s) under this subject folder."
  if [[ ${#T1S[@]} -gt 1 ]]; then
    printf '       %s\n' "${T1S[@]}"
  fi

  for T1 in "${T1S[@]}"; do
    # --------- CHANGED LOGIC STARTS HERE ----------
    # Mirror hierarchy: OUT_ROOT/<relative path to folder containing T1>
    if [[ "$T1" != "$IN_ROOT/"* ]]; then
      echo "[ERROR] T1 is not under IN_ROOT; skip: $T1" >&2
      FAILED=$((FAILED + 1))
      continue
    fi

    REL_PATH="${T1#$IN_ROOT/}"          # e.g., subject001/scan2/T1.nii.gz
    REL_DIR="$(dirname "$REL_PATH")"    # e.g., subject001/scan2
    SD="$OUT_ROOT/$REL_DIR"             # mirrored output directory for this scan
    SID="$FS_SID"                       # constant inside each SD
    OUT_SUB="$SD"                       # where we place exported .nii.gz
    MRI_DIR="$SD/$SID/mri"              # FastSurfer writes into --sd/--sid/mri
    # --------- CHANGED LOGIC ENDS HERE ----------

    echo ""
    echo "------------------------------------------------------------"
    echo "[INFO] T1 path:        $T1"
    echo "[INFO] mirrored OUT:   $OUT_SUB"
    echo "[INFO] fastsurfer sd:  $SD"
    echo "[INFO] fastsurfer sid: $SID"
    echo "------------------------------------------------------------"

    if [[ "$SKIP_DONE" == "1" && -f "$OUT_SUB/aseg.nii.gz" && -f "$OUT_SUB/aparc_aseg.nii.gz" ]]; then
      echo "[SKIP] outputs already exist"
      SKIPPED_DONE=$((SKIPPED_DONE + 1))
      continue
    fi

    mkdir -p "$SD"

    echo "[INFO] running FastSurfer (seg_only, GPU)..."
    if ! "$CTR" exec --nv --no-home -e \
        -B "$BIND_PATHS" \
        "$SIF_IMAGE" \
        /fastsurfer/run_fastsurfer.sh \
        --t1 "$T1" \
        --sid "$SID" \
        --sd "$SD" \
        --device cuda \
        --threads "$THREADS" \
        --seg_only \
        --no_cereb \
        --no_hypothal; then
      echo "[ERROR] FastSurfer failed for $SD (T1=$T1)" >&2
      FAILED=$((FAILED + 1))
      continue
    fi

    [[ -d "$MRI_DIR" ]] || { echo "[ERROR] missing MRI dir: $MRI_DIR" >&2; FAILED=$((FAILED + 1)); continue; }

    # FastSurfer seg_only outputs (under SD/SID/mri)
    ORIG_MGZ="$(pick_first_existing "$MRI_DIR/orig.mgz")" || { echo "[ERROR] missing orig.mgz" >&2; FAILED=$((FAILED + 1)); continue; }
    MASK_MGZ="$(pick_first_existing "$MRI_DIR/mask.mgz" "$MRI_DIR/brainmask.mgz")" || { echo "[ERROR] missing mask/brainmask mgz" >&2; FAILED=$((FAILED + 1)); continue; }
    ASEG_MGZ="$(pick_first_existing "$MRI_DIR/aseg.auto_noCCseg.mgz" "$MRI_DIR/aseg.mgz")" || { echo "[ERROR] missing aseg mgz" >&2; FAILED=$((FAILED + 1)); continue; }
    APARC_MGZ="$(pick_first_existing \
        "$MRI_DIR/aparc.DKTatlas+aseg.deep.mgz" \
        "$MRI_DIR/aparc.DKTatlas+aseg.mapped.mgz" \
        "$MRI_DIR/aparc+aseg.mgz")" || { echo "[ERROR] missing aparc+aseg mgz" >&2; FAILED=$((FAILED + 1)); continue; }

    echo "[INFO] exporting NIfTI (.nii.gz) via mri_convert (license bound into container)..."
    convert_mgz_to_nii "$ORIG_MGZ"  "$OUT_SUB/T1_fs_orig.nii.gz"
    convert_mgz_to_nii "$MASK_MGZ"  "$OUT_SUB/brainmask.nii.gz"
    convert_mgz_to_nii "$ASEG_MGZ"  "$OUT_SUB/aseg.nii.gz"
    convert_mgz_to_nii "$APARC_MGZ" "$OUT_SUB/aparc_aseg.nii.gz"

    DONE=$((DONE + 1))
    echo "[OK] finished $OUT_SUB"
  done
done

echo ""
echo "==================== SUMMARY ===================="
echo "[INFO] task:             $((TASK_ID + 1))/$N_SPLITS"
echo "[INFO] assigned folders: $ASSIGNED_SUBJECTS"
echo "[INFO] discovered T1s:   $DISCOVERED_T1"
echo "[INFO] done:             $DONE"
echo "[INFO] skipped done:     $SKIPPED_DONE"
echo "[INFO] skipped no T1:    $SKIPPED_NO_T1"
echo "[INFO] failed:           $FAILED"
echo "[INFO] finished at:      $(date)"
echo "================================================="
