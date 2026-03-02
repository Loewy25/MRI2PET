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

# =========================
# edit these lines if needed
# =========================
IN_ROOT="${IN_ROOT:-/scratch/l.peiwang/DIAN_geom}"
OUT_ROOT="${OUT_ROOT:-/scratch/l.peiwang/fastsurfer_simple_out}"
SIF_IMAGE="${SIF_IMAGE:-/scratch/l.peiwang/fastsurfer-gpu.sif}"
IMAGE_SRC="${IMAGE_SRC:-docker://deepmi/fastsurfer:latest}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SKIP_DONE="${SKIP_DONE:-1}"

# FastSurfer args kept intentionally simple
FS_EXTRA_ARGS=(--native_image --no_cereb --no_hypothal --no_cc)

pick_first_existing() {
    local p
    for p in "$@"; do
        if [[ -f "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

# -------------------------
# load apptainer/singularity
# -------------------------
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

mkdir -p "$OUT_ROOT"

# build container once if missing
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "[INFO] SIF not found, building: $SIF_IMAGE"
    echo "[INFO] source image: $IMAGE_SRC"
    "$CTR" build "$SIF_IMAGE" "$IMAGE_SRC"
fi

# bind only what the container needs to see
BIND_PATHS="$IN_ROOT:$IN_ROOT,$OUT_ROOT:$OUT_ROOT"

echo "[INFO] runtime:   $CTR"
echo "[INFO] IN_ROOT:   $IN_ROOT"
echo "[INFO] OUT_ROOT:  $OUT_ROOT"
echo "[INFO] SIF_IMAGE: $SIF_IMAGE"
echo "[INFO] THREADS:   $THREADS"
echo "[INFO] starting:  $(date)"

declare -a SUBJECT_DIRS=()
shopt -s nullglob
for d in "$IN_ROOT"/*; do
    [[ -d "$d" ]] || continue
    SUBJECT_DIRS+=("$d")
done
shopt -u nullglob

NUM_SUBS=${#SUBJECT_DIRS[@]}
if [[ "$NUM_SUBS" -eq 0 ]]; then
    echo "[FATAL] no subject subfolders found directly under $IN_ROOT" >&2
    exit 1
fi

echo "[INFO] found $NUM_SUBS subject folders under $IN_ROOT"

DONE=0
SKIPPED_DONE=0
SKIPPED_NO_T1=0
FAILED=0

for SUB in "${SUBJECT_DIRS[@]}"; do
    SUB_NAME="$(basename "$SUB")"
    SID="$(echo "$SUB_NAME" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
    OUT_SUB="$OUT_ROOT/$SID"
    MRI_DIR="$OUT_SUB/mri"

    mapfile -t T1S < <(find -L "$SUB" -type f -name 'T1.nii.gz' | sort)

    echo ""
    echo "============================================================"
    echo "[INFO] subject folder: $SUB"
    echo "[INFO] subject id:     $SID"
    echo "============================================================"

    if [[ ${#T1S[@]} -eq 0 ]]; then
        echo "[WARN] no T1.nii.gz found anywhere inside $SUB"
        SKIPPED_NO_T1=$((SKIPPED_NO_T1 + 1))
        continue
    fi

    if [[ ${#T1S[@]} -gt 1 ]]; then
        echo "[WARN] multiple T1.nii.gz files found under $SUB"
        printf '       %s\n' "${T1S[@]}"
        echo "[WARN] using the first one only"
    fi

    T1="${T1S[0]}"

    echo "[INFO] chosen T1:      $T1"
    echo "[INFO] output folder:  $OUT_SUB"

    if [[ "$SKIP_DONE" == "1" && -f "$OUT_SUB/aseg.nii.gz" && -f "$OUT_SUB/aparc_aseg.nii.gz" ]]; then
        echo "[SKIP] outputs already exist"
        SKIPPED_DONE=$((SKIPPED_DONE + 1))
        continue
    fi

    mkdir -p "$OUT_SUB"

    echo "[INFO] running FastSurfer..."
    if ! "$CTR" exec --nv --no-mount home,cwd -e \
        -B "$BIND_PATHS" \
        "$SIF_IMAGE" \
        /fastsurfer/run_fastsurfer.sh \
        --t1 "$T1" \
        --sid "$SID" \
        --sd "$OUT_ROOT" \
        --device cuda \
        --threads "$THREADS" \
        --seg_only \
        "${FS_EXTRA_ARGS[@]}"; then
        echo "[ERROR] FastSurfer failed for $SID" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    if [[ ! -d "$MRI_DIR" ]]; then
        echo "[ERROR] missing MRI dir after run: $MRI_DIR" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    ORIG_MGZ="$(pick_first_existing "$MRI_DIR/orig.mgz")" || {
        echo "[ERROR] missing orig.mgz for $SID" >&2
        FAILED=$((FAILED + 1))
        continue
    }
    MASK_MGZ="$(pick_first_existing "$MRI_DIR/mask.mgz" "$MRI_DIR/brainmask.mgz")" || {
        echo "[ERROR] missing mask/brainmask mgz for $SID" >&2
        FAILED=$((FAILED + 1))
        continue
    }
    ASEG_MGZ="$(pick_first_existing "$MRI_DIR/aseg.auto_noCCseg.mgz" "$MRI_DIR/aseg.mgz")" || {
        echo "[ERROR] missing aseg mgz for $SID" >&2
        FAILED=$((FAILED + 1))
        continue
    }
    APARC_MGZ="$(pick_first_existing "$MRI_DIR/aparc.DKTatlas+aseg.deep.mgz" "$MRI_DIR/aparc.DKTatlas+aseg.mapped.mgz" "$MRI_DIR/aparc+aseg.mgz")" || {
        echo "[ERROR] missing aparc+aseg mgz for $SID" >&2
        FAILED=$((FAILED + 1))
        continue
    }

    echo "[INFO] exporting NIfTI files..."
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$ORIG_MGZ"  "$OUT_SUB/T1_fs_orig.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$MASK_MGZ"  "$OUT_SUB/brainmask.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$ASEG_MGZ"  "$OUT_SUB/aseg.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$APARC_MGZ" "$OUT_SUB/aparc_aseg.nii.gz"

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
