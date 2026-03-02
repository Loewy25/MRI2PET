#!/bin/bash
#SBATCH --job-name=fastsurfer_seg
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=fs.out
#SBATCH --error=fs.err

set -euo pipefail

# =========================
# edit these 4 lines only
# =========================
IN_ROOT="${IN_ROOT:-/scratch/l.peiwang/DIAN_geom}"
OUT_ROOT="${OUT_ROOT:-/scratch/l.peiwang/fastsurfer_simple_out}"
SIF_IMAGE="${SIF_IMAGE:-/scratch/l.peiwang/fastsurfer-gpu.sif}"
FASTSURFER_IMAGE_SOURCE="${FASTSURFER_IMAGE_SOURCE:-docker://deepmi/fastsurfer:cuda-v2.5.0}"

# optional
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SKIP_DONE="${SKIP_DONE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:---native_image --no_cereb --no_hypothal --no_cc}"

mkdir -p "$OUT_ROOT"

# -------- load apptainer / singularity --------
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

echo "[INFO] container runtime: $CTR"
echo "[INFO] input root:        $IN_ROOT"
echo "[INFO] output root:       $OUT_ROOT"
echo "[INFO] sif image:         $SIF_IMAGE"
echo "[INFO] extra args:        $EXTRA_ARGS"

# -------- build image once if missing --------
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "[INFO] building FastSurfer image: $SIF_IMAGE"
    "$CTR" build "$SIF_IMAGE" "$FASTSURFER_IMAGE_SOURCE"
fi

# -------- bind only what we need --------
BIND_PATHS="$IN_ROOT:$IN_ROOT,$OUT_ROOT:$OUT_ROOT"

# -------- collect all T1 files --------
mapfile -t T1S < <(find "$IN_ROOT" -type f -name 'T1.nii.gz' | sort)
N=${#T1S[@]}

echo "[INFO] found $N T1 files"
if [[ "$N" -eq 0 ]]; then
    echo "[FATAL] no T1.nii.gz found under $IN_ROOT" >&2
    exit 1
fi

# -------- loop subjects --------
DONE=0
SKIPPED=0
FAILED=0

for T1 in "${T1S[@]}"; do
    REL_DIR="${T1#$IN_ROOT/}"
    REL_DIR="$(dirname "$REL_DIR")"
    SID="$(echo "$REL_DIR" | sed 's#/#__#g')"

    SUBJECT_DIR="$OUT_ROOT/$SID"
    MRI_DIR="$SUBJECT_DIR/mri"

    echo ""
    echo "============================================================"
    echo "[INFO] subject: $SID"
    echo "[INFO] T1:      $T1"
    echo "[INFO] out:     $SUBJECT_DIR"
    echo "============================================================"

    if [[ "$SKIP_DONE" == "1" && -f "$SUBJECT_DIR/aparc_aseg.nii.gz" && -f "$SUBJECT_DIR/aseg.nii.gz" ]]; then
        echo "[SKIP] outputs already exist"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    mkdir -p "$SUBJECT_DIR"

    if "$CTR" exec --nv --no-mount home,cwd -e \
        -B "$BIND_PATHS" \
        "$SIF_IMAGE" \
        /fastsurfer/run_fastsurfer.sh \
        --t1 "$T1" \
        --sid "$SID" \
        --sd "$OUT_ROOT" \
        --device cuda \
        --threads "$THREADS" \
        --seg_only \
        $EXTRA_ARGS
    then
        echo "[INFO] segmentation finished for $SID"
    else
        echo "[ERROR] FastSurfer failed for $SID" >&2
        FAILED=$((FAILED+1))
        continue
    fi

    if [[ ! -d "$MRI_DIR" ]]; then
        echo "[ERROR] missing mri dir: $MRI_DIR" >&2
        FAILED=$((FAILED+1))
        continue
    fi

    # export the main outputs to nii.gz
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$MRI_DIR/orig.mgz" "$SUBJECT_DIR/T1_fs_orig.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$MRI_DIR/mask.mgz" "$SUBJECT_DIR/brainmask.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$MRI_DIR/aseg.auto_noCCseg.mgz" "$SUBJECT_DIR/aseg.nii.gz"
    "$CTR" exec -B "$BIND_PATHS" "$SIF_IMAGE" mri_convert "$MRI_DIR/aparc.DKTatlas+aseg.deep.mgz" "$SUBJECT_DIR/aparc_aseg.nii.gz"

    DONE=$((DONE+1))
    echo "[OK] exported NIfTI files for $SID"
done

echo ""
echo "==================== SUMMARY ===================="
echo "[INFO] done:    $DONE"
echo "[INFO] skipped: $SKIPPED"
echo "[INFO] failed:  $FAILED"
echo "[INFO] finished at: $(date)"
echo "================================================="


echo "[OK] Exported: $OUT"
echo "[INFO] Finished: $(date)"
