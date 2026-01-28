#!/bin/bash
#SBATCH --job-name=DIAN_geom
#SBATCH --mem=10G
#SBATCH -t 23:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --output=slurm-%A_geom.out
#SBATCH --error=slurm-%A_geom.out

set -euo pipefail

IN_ROOT="/scratch/l.peiwang/DIAN"
OUT_ROOT="/scratch/l.peiwang/DIAN_geom"
mkdir -p "$OUT_ROOT"

MIN_VOXELS=15000000
MAX_ANISO=1.25

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} user=$USER"
echo "[INFO] IN_ROOT=$IN_ROOT"
echo "[INFO] OUT_ROOT=$OUT_ROOT"

# ---------- Robust module init (without sourcing /etc/profile) ----------
# Temporarily disable nounset while sourcing env scripts that may reference unset vars.
set +u
for f in /etc/profile.d/modules.sh \
         /usr/share/Modules/init/bash \
         /usr/share/lmod/lmod/init/bash; do
  [[ -r "$f" ]] && source "$f" || true
done
set -u

# Try module load FSL (if module exists)
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load fsl/6.0.7.8 >/dev/null 2>&1 || true
  module list 2>/dev/null || true
else
  echo "[WARN] module command not available in this shell"
fi

USE_FSLINFO=0
if command -v fslinfo >/dev/null 2>&1; then
  USE_FSLINFO=1
  echo "[INFO] using fslinfo: $(command -v fslinfo)"
else
  echo "[WARN] fslinfo not found; will try Python nibabel fallback"
fi

py_hdr() {
python3 - "$1" <<'PY'
import sys
nii = sys.argv[1]
try:
    import nibabel as nib
except Exception as e:
    print("NO_NIBABEL", e)
    sys.exit(3)
img = nib.load(nii)
hdr = img.header
d1,d2,d3 = (int(hdr['dim'][1]), int(hdr['dim'][2]), int(hdr['dim'][3]))
p1,p2,p3 = (float(hdr['pixdim'][1]), float(hdr['pixdim'][2]), float(hdr['pixdim'][3]))
print(d1, d2, d3, p1, p2, p3)
PY
}

read_geom() {
  local nii="$1"
  if [[ "$USE_FSLINFO" -eq 1 ]]; then
    fslinfo "$nii" | awk '
      $1=="dim1"{d1=$2}
      $1=="dim2"{d2=$2}
      $1=="dim3"{d3=$2}
      $1=="pixdim1"{p1=$2}
      $1=="pixdim2"{p2=$2}
      $1=="pixdim3"{p3=$2}
      END{print d1,d2,d3,p1,p2,p3}'
  else
    py_hdr "$nii"
  fi
}

pass=0
fail=0
sess_fail=0

for sess in "$IN_ROOT"/*_mr; do
  [[ -d "$sess" ]] || continue
  sess_id="$(basename "$sess")"
  out_sess="$OUT_ROOT/$sess_id"
  mkdir -p "$out_sess"

  summary="$out_sess/geom_summary.tsv"
  echo -e "series\tvoxels\tdims\tpixdims\taniso\tstatus\treason" > "$summary"

  any_pass=0

  while IFS= read -r nii; do
    series="$(basename "$(dirname "$nii")")"

    geom="$(read_geom "$nii" || true)"
    if [[ "$geom" == NO_NIBABEL* ]]; then
      echo "[FATAL] nibabel not available and fslinfo missing. Load FSL correctly or install nibabel."
      echo "[DEBUG] $geom"
      exit 4
    fi

    read -r d1 d2 d3 p1 p2 p3 <<<"$geom"

    voxels="$(awk -v a="$d1" -v b="$d2" -v c="$d3" 'BEGIN{printf "%.0f",a*b*c}')"
    aniso="$(awk -v a="$p1" -v b="$p2" -v c="$p3" 'BEGIN{
      min=a; if(b<min)min=b; if(c<min)min=c;
      max=a; if(b>max)max=b; if(c>max)max=c;
      printf "%.4f",max/(min+1e-12)
    }')"

    status="PASS"; reason="ok"
    if [[ "$voxels" -lt "$MIN_VOXELS" ]]; then
      status="FAIL"; reason="small_volume"
    elif awk -v x="$aniso" -v y="$MAX_ANISO" 'BEGIN{exit (x<=y)?0:1}'; then
      :
    else
      status="FAIL"; reason="anisotropic"
    fi

    echo -e "${series}\t${voxels}\t${d1}x${d2}x${d3}\t${p1}x${p2}x${p3}\t${aniso}\t${status}\t${reason}" >> "$summary"

    if [[ "$status" == "PASS" ]]; then
      any_pass=1
      mkdir -p "$out_sess/$series"
      ln -sfn "$nii" "$out_sess/$series/T1.nii.gz"
      [[ -f "$(dirname "$nii")/T1.json" ]] && ln -sfn "$(dirname "$nii")/T1.json" "$out_sess/$series/T1.json" || true
      ((pass+=1))
    else
      ((fail+=1))
    fi

  done < <(find "$sess" -mindepth 2 -maxdepth 2 -type f -name "T1.nii.gz" | sort)

  if [[ "$any_pass" -eq 0 ]]; then
    echo "[SESSION_FAIL] $sess_id : no candidates passed geometry (see $summary)"
    ((sess_fail+=1))
  else
    echo "[SESSION_OK]   $sess_id : done (see $summary)"
  fi
done

echo
echo "Done."
echo "candidate_pass=$pass candidate_fail=$fail sessions_with_no_pass=$sess_fail"
echo "Thresholds: MIN_VOXELS=$MIN_VOXELS MAX_ANISO=$MAX_ANISO"


