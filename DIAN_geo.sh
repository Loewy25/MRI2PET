#!/bin/bash
#SBATCH --job-name=MGDA_UB
#SBATCH --mem=10G
#SBATCH -t 23:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --output=slurm-%A_haha.out
#SBATCH --error=slurm-%A_haha.out
set -euo pipefail

module load fsl

# INPUT: already-converted candidates
IN_ROOT="/scratch/l.peiwang/DIAN"

# OUTPUT: geometry-filtered staging area
OUT_ROOT="/scratch/l.peiwang/DIAN_geom"
mkdir -p "$OUT_ROOT"

# geometry thresholds (industry-safe defaults)
MIN_VOXELS=15000000
MAX_ANISO=1.25

if ! command -v fslinfo >/dev/null 2>&1; then
  echo "[FATAL] fslinfo not found. Load FSL."
  exit 2
fi

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

    # extract geometry
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

    echo -e "${series}\t${voxels}\t${d1}x${d2}x${d3}\t${p1}x${p2}x${p3}\t${aniso}\t${status}\t${reason}" \
      >> "$summary"

    if [[ "$status" == "PASS" ]]; then
      any_pass=1
      mkdir -p "$out_sess/$series"
      ln -sfn "$nii" "$out_sess/$series/T1.nii.gz"
      [[ -f "$(dirname "$nii")/T1.json" ]] && \
        ln -sfn "$(dirname "$nii")/T1.json" "$out_sess/$series/T1.json"
      ((pass+=1))
    else
      ((fail+=1))
    fi

  done < <(find "$in_sess" -mindepth 2 -maxdepth 2 -type f -name "T1.nii.gz" | sort)

  if [[ "$any_pass" -eq 0 ]]; then
    echo "[SESSION_FAIL] $sess_id : no candidates passed geometry"
    ((sess_fail+=1))
  else
    echo "[SESSION_OK]   $sess_id : geometry filtering complete"
  fi
done

echo
echo "Done."
echo "candidate_pass=$pass"
echo "candidate_fail=$fail"
echo "sessions_with_no_pass=$sess_fail"
echo "IN_ROOT=$IN_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "Thresholds: MIN_VOXELS=$MIN_VOXELS  MAX_ANISO=$MAX_ANISO"
