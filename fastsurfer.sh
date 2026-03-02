#!/bin/bash
#SBATCH --job-name=DIAN_FastSurfer
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Default broad array so plain "sbatch DIAN_fastsurfer.sh" works.
# Tasks beyond available inputs auto-exit in-script.
#SBATCH --array=1-5000%10
#SBATCH --output=slurm-%A_%a_fastsurfer.out
#SBATCH --error=slurm-%A_%a_fastsurfer.out

set -euo pipefail

# ---------------- Robust module init ----------------
set +u
for f in /etc/profile.d/modules.sh \
         /usr/share/Modules/init/bash \
         /usr/share/lmod/lmod/init/bash; do
  [[ -r "$f" ]] && source "$f" || true
done
set -u

# ---------------- Configurable paths ----------------
IN_ROOT="${IN_ROOT:-/scratch/l.peiwang/DIAN_geom}"  # input candidates folder
LIST="${LIST:-}"                                     # optional TSV (helper_DIAN format)
SUBJECTS_DIR="${SUBJECTS_DIR:-/scratch/l.peiwang/fastsurfer_DIAN}"
FINAL_ROOT="${FINAL_ROOT:-${EXPORT_ROOT:-/scratch/l.peiwang/DIAN_fastsurfer_final}}"
EXPORT_ROOT="$FINAL_ROOT"
FS_LICENSE="${FS_LICENSE:-/ceph/chpc/mapped/brier/software/freesurfer/license.txt}"

# ---------------- Optional module names ----------------
FASTSURFER_MODULE="${FASTSURFER_MODULE:-fastsurfer}"
FREESURFER_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}"

# ---------------- FastSurfer behavior ----------------
FASTSURFER_RUNNER="${FASTSURFER_RUNNER:-run_fastsurfer.sh}"
FASTSURFER_DEVICE="${FASTSURFER_DEVICE:-cuda}"   # cuda or cpu
FASTSURFER_MODE="${FASTSURFER_MODE:-seg_only}"   # seg_only or full
FASTSURFER_EXTRA_ARGS="${FASTSURFER_EXTRA_ARGS:-}"
FASTSURFER_PY="${FASTSURFER_PY:-}"               # optional python executable passed via --py
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

die() {
  echo "[FATAL] $*" >&2
  exit 2
}

warn() {
  echo "[WARN] $*" >&2
}

cleanup() {
  [[ -n "${TMP_LIST:-}" && -f "${TMP_LIST:-}" ]] && rm -f "$TMP_LIST"
}

pick_first_existing() {
  local p
  for p in "$@"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
  return 1
}

resolve_runner() {
  local cand
  cand="$FASTSURFER_RUNNER"
  if [[ -x "$cand" ]]; then
    echo "$cand"
    return 0
  fi
  if command -v "$cand" >/dev/null 2>&1; then
    command -v "$cand"
    return 0
  fi
  if [[ -n "${FASTSURFER_HOME:-}" && -x "$FASTSURFER_HOME/run_fastsurfer.sh" ]]; then
    echo "$FASTSURFER_HOME/run_fastsurfer.sh"
    return 0
  fi
  return 1
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

need_fs_license() {
  # Surface pipeline requires a FreeSurfer license.
  if [[ "$FASTSURFER_MODE" == "full" ]]; then
    return 0
  fi
  # Segmentation-only may still need license for talairach or T1/T2 registration.
  if [[ "$FASTSURFER_EXTRA_ARGS" == *"--tal_reg"* ]]; then
    return 0
  fi
  if [[ "$FASTSURFER_EXTRA_ARGS" == *"--t2 "* ]] && [[ "$FASTSURFER_EXTRA_ARGS" != *"--reg_mode none"* ]]; then
    return 0
  fi
  return 1
}

build_list_from_in_root() {
  local in_root="$1"
  local out_list="$2"
  local nii_link series sess_id target

  echo -e "sess_id\tseries\tnii_link\ttarget" > "$out_list"

  while IFS= read -r nii_link; do
    [[ -n "$nii_link" ]] || continue
    series="$(basename "$(dirname "$nii_link")")"
    sess_id="$(basename "$(dirname "$(dirname "$nii_link")")")"
    target="$(readlink -f "$nii_link" 2>/dev/null || true)"
    [[ -n "$target" ]] || target="$nii_link"
    echo -e "${sess_id}\t${series}\t${nii_link}\t${target}" >> "$out_list"
  done < <(find -L "$in_root" -type f -name "T1.nii.gz" | sort)
}

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load "$FASTSURFER_MODULE" >/dev/null 2>&1 || true
  module load "$FREESURFER_MODULE" >/dev/null 2>&1 || true
fi

RUNNER="$(resolve_runner || true)"
[[ -n "$RUNNER" ]] || die "cannot find FastSurfer runner. Set FASTSURFER_RUNNER or FASTSURFER_HOME."

if [[ -z "$FASTSURFER_PY" ]]; then
  if [[ -n "${FASTSURFER_HOME:-}" && -x "$FASTSURFER_HOME/.venv/bin/python3" ]]; then
    FASTSURFER_PY="$FASTSURFER_HOME/.venv/bin/python3"
  elif [[ -n "${FASTSURFER_HOME:-}" && -x "$FASTSURFER_HOME/.venv/bin/python" ]]; then
    FASTSURFER_PY="$FASTSURFER_HOME/.venv/bin/python"
  fi
fi

TMP_LIST=""
trap cleanup EXIT

if [[ -n "$LIST" ]]; then
  [[ -f "$LIST" ]] || die "LIST is set but file does not exist: $LIST"
  INPUT_LIST="$LIST"
else
  [[ -d "$IN_ROOT" ]] || die "IN_ROOT does not exist: $IN_ROOT"
  TMP_LIST="$(mktemp /tmp/dian_geom_all_t1_list.XXXXXX.tsv)"
  build_list_from_in_root "$IN_ROOT" "$TMP_LIST"
  INPUT_LIST="$TMP_LIST"
fi

[[ -f "$INPUT_LIST" ]] || die "missing input list: $INPUT_LIST"

task_id="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
[[ -n "${task_id}" ]] || die "SLURM_ARRAY_TASK_ID is not set (or provide TASK_ID)."

line="$(awk -v n="$task_id" 'NR==n+1{print}' "$INPUT_LIST")"
[[ -n "$line" ]] || { echo "[INFO] no list entry for task_id=$task_id"; exit 0; }

sess_id="$(echo "$line" | cut -f1)"
series="$(echo "$line" | cut -f2)"
nii="$(echo "$line" | cut -f4)"
[[ -f "$nii" ]] || die "input T1 does not exist: $nii"

series_safe="$(echo "$series" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
FSID="${sess_id}__${series_safe}"
OUT="$EXPORT_ROOT/$FSID"

mkdir -p "$SUBJECTS_DIR" "$OUT"
export SUBJECTS_DIR
license_required=0
if need_fs_license; then
  license_required=1
fi

if (( license_required )); then
  [[ -f "$FS_LICENSE" ]] || die "FreeSurfer license required but missing: $FS_LICENSE"
  export FS_LICENSE
else
  if [[ -f "$FS_LICENSE" ]]; then
    export FS_LICENSE
  else
    warn "FS_LICENSE not found at $FS_LICENSE; proceeding in seg_only mode."
  fi
fi

if [[ "$SKIP_IF_DONE" == "1" ]] \
  && [[ -s "$OUT/T1_fs_orig.nii.gz" ]] \
  && [[ -s "$OUT/brainmask.nii.gz" ]] \
  && [[ -s "$OUT/aseg.nii.gz" ]] \
  && [[ -s "$OUT/aparc_aseg.nii.gz" ]]; then
  echo "[SKIP] FSID=$FSID already exported."
  exit 0
fi

echo "[INFO] FSID=$FSID"
echo "[INFO] T1=$nii"
echo "[INFO] FASTSURFER_MODE=$FASTSURFER_MODE FASTSURFER_DEVICE=$FASTSURFER_DEVICE"
echo "[INFO] Runner=$RUNNER"
echo "[INFO] INPUT_LIST=$INPUT_LIST"
echo "[INFO] EXPORT_ROOT=$EXPORT_ROOT"
echo "[INFO] Start: $(date)"

cmd=(
  "$RUNNER"
  --t1 "$nii"
  --sid "$FSID"
  --sd "$SUBJECTS_DIR"
  --device "$FASTSURFER_DEVICE"
  --threads "${SLURM_CPUS_PER_TASK:-4}"
)

if [[ -f "$FS_LICENSE" ]]; then
  cmd+=(--fs_license "$FS_LICENSE")
fi

if [[ -n "$FASTSURFER_PY" ]]; then
  cmd+=(--py "$FASTSURFER_PY")
fi

case "$FASTSURFER_MODE" in
  seg_only) cmd+=(--seg_only) ;;
  full) ;;
  *) die "FASTSURFER_MODE must be seg_only or full (got: $FASTSURFER_MODE)" ;;
esac

if [[ -n "$FASTSURFER_EXTRA_ARGS" ]]; then
  # Intentionally split FASTSURFER_EXTRA_ARGS on spaces (e.g., "--3T --no_hypothal").
  # shellcheck disable=SC2206
  extra=( $FASTSURFER_EXTRA_ARGS )
  cmd+=("${extra[@]}")
fi

echo "[INFO] Running FastSurfer command:"
echo "       ${cmd[*]}"
"${cmd[@]}"

mri_dir="$SUBJECTS_DIR/$FSID/mri"
[[ -d "$mri_dir" ]] || die "missing output dir: $mri_dir"

orig_src="$(pick_first_existing "$mri_dir/orig.mgz")" \
  || die "missing orig mgz in $mri_dir"
brainmask_src="$(pick_first_existing "$mri_dir/brainmask.mgz" "$mri_dir/mask.mgz")" \
  || die "missing brainmask output in $mri_dir"
aseg_src="$(pick_first_existing "$mri_dir/aseg.mgz" "$mri_dir/aseg.auto_noCCseg.mgz")" \
  || die "missing aseg output in $mri_dir"
aparc_src="$(pick_first_existing \
  "$mri_dir/aparc+aseg.mgz" \
  "$mri_dir/aparc.DKTatlas+aseg.mgz" \
  "$mri_dir/aparc.DKTatlas+aseg.mapped.mgz" \
  "$mri_dir/aparc.DKTatlas+aseg.deep.withCC.mgz" \
  "$mri_dir/aparc.DKTatlas+aseg.deep.mgz")" \
  || die "missing aparc+aseg output in $mri_dir"

convert_mgz_to_nii() {
  local src="$1"
  local dst="$2"
  if command -v mri_convert >/dev/null 2>&1; then
    mri_convert "$src" "$dst"
    return 0
  fi
  python3 - "$src" "$dst" <<'PY'
import sys
try:
    import nibabel as nib
except Exception as e:
    print(f"[FATAL] nibabel import failed: {e}", file=sys.stderr)
    sys.exit(3)
src, dst = sys.argv[1], sys.argv[2]
img = nib.load(src)
nib.save(img, dst)
PY
}

convert_mgz_to_nii "$orig_src"      "$OUT/T1_fs_orig.nii.gz"
convert_mgz_to_nii "$brainmask_src" "$OUT/brainmask.nii.gz"
convert_mgz_to_nii "$aseg_src"      "$OUT/aseg.nii.gz"
convert_mgz_to_nii "$aparc_src"     "$OUT/aparc_aseg.nii.gz"

{
  echo -e "output\tinput"
  echo -e "T1_fs_orig.nii.gz\t$orig_src"
  echo -e "brainmask.nii.gz\t$brainmask_src"
  echo -e "aseg.nii.gz\t$aseg_src"
  echo -e "aparc_aseg.nii.gz\t$aparc_src"
} > "$OUT/export_sources.tsv"

echo "[OK] Exported: $OUT"
echo "[INFO] Finished: $(date)"
