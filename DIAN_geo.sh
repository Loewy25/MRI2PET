#!/bin/bash
#SBATCH --job-name=DIAN_geom
#SBATCH --partition=tier1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=geom_%j.out
#SBATCH --error=geom_%j.err

set -euo pipefail

# ===================== USER PATHS =====================
IN_ROOT="/scratch/l.peiwang/DIAN"        # has <session>/<series>/T1.nii.gz
OUT_ROOT="/scratch/l.peiwang/DIAN_geom"  # new root to write filtered results
mkdir -p "$OUT_ROOT"

# ===================== THRESHOLDS =====================
MIN_VOXELS=15000000
MAX_ANISO=1.25

# ===================== FIND/ENABLE FSLINFO =====================
echo "[INFO] host=$(hostname)  pwd=$(pwd)"
echo "[INFO] bash=$BASH_VERSION"

# Try to initialize environment/module system (harmless if absent)
for f in /etc/profile \
         /etc/profile.d/modules.sh \
         /usr/share/lmod/lmod/init/bash \
         /usr/share/Modules/init/bash; do
  [[ -r "$f" ]] && source "$f" || true
done

# Try module load if module exists
if command -v module >/dev/null 2>&1; then
  echo "[INFO] module system detected: $(command -v module)"
  module load fsl/6.0.7.8 >/dev/null 2>&1 || true
  module list 2>/dev/null || true
else
  echo "[INFO] module command not available in this job shell"
fi

# If still missing, try to locate fslinfo and prepend PATH
if ! command -v fslinfo >/dev/null 2>&1; then
  echo "[WARN] fslinfo not in PATH after module load; attempting bounded search..."

  found=""
  for base in /usr/local /opt /ceph /share /cvmfs; do
    [[ -d "$base" ]] || continue
    found="$(find "$base" -maxdepth 6 -type f -name fslinfo -perm -111 2>/dev/null | head -n 1 || true)"
    [[ -n "$found" ]] && break
  done

  if [[ -n "$found" ]]; then
    bindir="$(dirname "$found")"
    export PATH="$bindir:$PATH"
    echo "[INFO] found fslinfo at: $found"
  fi
fi

# Final check
if ! command -v fslinfo >/dev/null 2>&1; then
  echo "[FATAL] fslinfo still not found."
  echo "[DEBUG] PATH=$PATH"
  echo "[DEBUG] which module: $(command -v module || echo NONE)"
  exit 2
fi

echo "[INFO] using fslinfo: $(command -v fslinfo)"

# ===================== GEOMETRY FILTER =====================
pass=0
fail=0
sess_fail=0

for sess in "$IN_ROOT"/*_mr; do
  [[ -d "$sess" ]] || continue
  sess_id="$(basename "$sess")"

  in_sess="$sess"
  out_sess="$OUT_ROOT/$sess_id"
  mkdir -p "$out_sess"

  summary="$out_sess/geom_summary.tsv"
  echo -e "series\tvoxels\tdims\tpixdims\taniso\tstatus\treason" > "$summary"

  any_pass=0

  while IFS= read -r nii; do
    series="$(basename "$(dirname "$nii")")"

    # Extract dims + pixdims from header
    read -r d1 d2 d3 p1 p2 p3 < <(
      fslinfo "$nii" | awk '
        $1=="dim1"{d1=$2}
        $1=="dim2"{d2=$2}
        $1=="dim3"{d3=$2}
        $1=="pixdim1"{p1=$2}
        $1=="pixdim2"{p2=$2}
        $1=="pixdim3"{p3=$2}
        END{printf "%s %s %s %s %s %s\n", d1,d2,d3,p1,p2,p3}'
    )

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

  done < <(find "$in_sess" -mindepth 2 -maxdepth 2 -type f -name "T1.nii.gz" | sort)

  if [[ "$any_pass" -eq 0 ]]; then
    echo "[SESSION_FAIL] $sess_id : no candidates passed geometry (see $summary)"
    ((sess_fail+=1))
  else
    echo "[SESSION_OK]   $sess_id : done (see $summary)"
  fi
done

echo
echo "Done."
echo "candidate_pass=$pass  candidate_fail=$fail  sessions_with_no_pass=$sess_fail"
echo "IN_ROOT=$IN_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "Thresholds: MIN_VOXELS=$MIN_VOXELS  MAX_ANISO=$MAX_ANISO"

