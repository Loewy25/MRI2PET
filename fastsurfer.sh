#!/bin/bash
#SBATCH --job-name=fastsurfer_seg
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j_fastsurfer.out
#SBATCH --error=slurm-%j_fastsurfer.out

set -euo pipefail

# ============================================================
# Config (override with: sbatch --export=ALL,IN_ROOT=...,OUT_ROOT=... script.sh)
# ============================================================
IN_ROOT="${IN_ROOT:-/scratch/l.peiwang/DIAN_geom}"                  # contains subject folders
OUT_ROOT="${OUT_ROOT:-/scratch/l.peiwang/fastsurfer_simple_out}"     # FastSurfer --sd
SIF_IMAGE="${SIF_IMAGE:-/scratch/l.peiwang/fastsurfer-gpu.sif}"      # built .sif
IMAGE_SRC="${IMAGE_SRC:-docker://deepmi/fastsurfer:latest}"          # for building sif if missing
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SKIP_DONE="${SKIP_DONE:-1}"
ONLY_SUBJECT="${ONLY_SUBJECT:-}"                                    # optional: run only one folder name

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

# Prefer a non-"repeat" T1 if multiple; else first sorted.
choose_t1() {
  local -n arr_ref=$1
  local f low
  local preferred=()
  for f in "${arr_ref[@]}"; do
    low="$(printf '%s' "$f" | tr '[:upper:]' '[:lower:]')"
    [[ "$low" == *repeat* ]] || preferred+=("$f")
  done
  if [[ ${#preferred[@]} -gt 0 ]]; then
    echo "${preferred[0]}"
  else
    echo "${arr_ref[0]}"
  fi
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

# Build SIF once if missing
if [[ ! -f "$SIF_IMAGE" ]]; then
  echo "[INFO] SIF missing, building: $SIF_IMAGE"
  echo "[INFO] from: $IMAGE_SRC"
  "$CTR" build "$SIF_IMAGE" "$IMAGE_SRC"
fi

# Avoid container warning about missing /home/...
cd /tmp

echo "[INFO] runtime:        $CTR"
echo "[INFO] IN_ROOT:        $IN_ROOT"
echo "[INFO] OUT_ROOT:       $OUT_ROOT"
echo "[INFO] SIF_IMAGE:      $SIF_IMAGE"
echo "[INFO] THREADS:        $THREADS"
echo "[INFO] BIND_PATHS:     $BIND_PATHS"
echo "[INFO] FS_LICENSE_HOST $FS_LICENSE_HOST -> $FS_LICENSE_IN (in container)"
echo "[INFO] ONLY_SUBJECT:   ${ONLY_SUBJECT:-<all>}"
echo "[INFO] start:          $(date)"

# ============================================================
# Enumerate subject folders (immediate children)
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

echo "[INFO] found ${#SUBJECT_DIRS[@]} subject folders."

DONE=0
SKIPPED_DONE=0
SKIPPED_NO_T1=0
FAILED=0

# ============================================================
# Main loop
# ============================================================
for SUB in "${SUBJECT_DIRS[@]}"; do
  SUB_NAME="$(basename "$SUB")"
  SID="$(echo "$SUB_NAME" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  OUT_SUB="$OUT_ROOT/$SID"
  MRI_DIR="$OUT_SUB/mri"

  echo ""
  echo "============================================================"
  echo "[INFO] subject folder: $SUB"
  echo "[INFO] subject id:     $SID"
  echo "============================================================"

  mapfile -t T1S < <(find -L "$SUB" -type f -name 'T1.nii.gz' | sort)

  if [[ ${#T1S[@]} -eq 0 ]]; then
    echo "[WARN] no T1.nii.gz found inside $SUB"
    SKIPPED_NO_T1=$((SKIPPED_NO_T1 + 1))
    continue
  fi

  if [[ ${#T1S[@]} -gt 1 ]]; then
    echo "[WARN] multiple T1.nii.gz found:"
    printf '       %s\n' "${T1S[@]}"
    echo "[WARN] choosing first non-repeat (else first sorted)"
  fi

  T1="$(choose_t1 T1S)"
  echo "[INFO] chosen T1:      $T1"
  echo "[INFO] output folder:  $OUT_SUB"

  if [[ "$SKIP_DONE" == "1" && -f "$OUT_SUB/aseg.nii.gz" && -f "$OUT_SUB/aparc_aseg.nii.gz" ]]; then
    echo "[SKIP] outputs already exist"
    SKIPPED_DONE=$((SKIPPED_DONE + 1))
    continue
  fi

  mkdir -p "$OUT_SUB"

  echo "[INFO] running FastSurfer (seg_only, GPU)..."
  if ! "$CTR" exec --nv --no-home -e \
      -B "$BIND_PATHS" \
      "$SIF_IMAGE" \
      /fastsurfer/run_fastsurfer.sh \
      --t1 "$T1" \
      --sid "$SID" \
      --sd "$OUT_ROOT" \
      --device cuda \
      --threads "$THREADS" \
      --seg_only \
      --no_cereb \
      --no_hypothal; then
    echo "[ERROR] FastSurfer failed for $SID" >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  [[ -d "$MRI_DIR" ]] || { echo "[ERROR] missing MRI dir: $MRI_DIR" >&2; FAILED=$((FAILED + 1)); continue; }

  # FastSurfer seg_only outputs
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
  echo "[OK] finished $SID"
done

echo ""
echo "==================== SUMMARY ===================="
echo "[INFO] done:            $DONE"
echo "[INFO] skipped done:    $SKIPPED_DONE"
echo "[INFO] skipped no T1:   $SKIPPED_NO_T1"
echo "[INFO] failed:          $FAILED"
echo "[INFO] finished at:     $(date)"
echo "================================================="
