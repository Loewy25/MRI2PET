#!/bin/bash
#SBATCH --job-name=obs_t1_dcm2niix
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=obs_t1_%j.out
#SBATCH --error=obs_t1_%j.err

set -euo pipefail

IN_ROOT="/ceph/chpc/mapped/dian_obs_data_shared/obs_mr_scans_imagids"
OUT_ROOT="/scratch/l.peiwang/DIAN"

mkdir -p "$OUT_ROOT"

# include: segmentation-grade 3D T1 family (multi-vendor)
INC_RE='(mprage|mp[-_ ]?rage|spgr|fspgr|ir[-_ ]?fspgr|bravo|tfe)'
# exclude: obvious non-anat / junk / other modalities
EXC_RE='(moco|mosaic|localizer|3[_ -]?plane|phoenix|zipreport|report|field|mag|pha|phase|swi|mip|flair|t2|dti|diff|adc|tracew|tensor|colfa|fa|fmri|rsfmri|rest|asl|perfusion|default_ps_series|surv_new_scale_parameters|unknown)'

missing=0
multi=0
converted=0
failed=0

sanitize() { echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

for sess_dir in "$IN_ROOT"/*_mr; do
  [[ -d "$sess_dir" ]] || continue
  sess="$(basename "$sess_dir")"
  out_sess="$OUT_ROOT/$sess"
  mkdir -p "$out_sess"

  mapfile -t cands < <(
    find "$sess_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
      | awk -v INC="$INC_RE" -v EXC="$EXC_RE" 'BEGIN{IGNORECASE=1}
          $0 ~ INC && $0 !~ EXC {print}'
      | sort
  )

  if [[ "${#cands[@]}" -eq 0 ]]; then
    echo "[MISSING] $sess : no T1 candidates (INC=$INC_RE)"
    ((missing+=1))
    continue
  fi
  if [[ "${#cands[@]}" -gt 1 ]]; then
    echo "[MULTI]   $sess : ${#cands[@]} candidates"
    ((multi+=1))
  else
    echo "[ONE]     $sess : 1 candidate"
  fi

  for series in "${cands[@]}"; do
    src="$sess_dir/$series"
    series_safe="$(sanitize "$series")"
    dst="$out_sess/$series_safe"
    mkdir -p "$dst"

    tmp="$dst/.tmp_dcm2niix"
    rm -rf "$tmp"; mkdir -p "$tmp"

    if ! dcm2niix -z y -f T1 -o "$tmp" "$src" >/dev/null 2>&1; then
      echo "[FAIL]    $sess | $series : dcm2niix error"
      rm -rf "$tmp"
      ((failed+=1))
      continue
    fi

    nii="$(ls -1t "$tmp"/T1*.nii.gz 2>/dev/null | head -n 1 || true)"
    jsn="$(ls -1t "$tmp"/T1*.json 2>/dev/null | head -n 1 || true)"
    if [[ -z "${nii:-}" ]]; then
      echo "[FAIL]    $sess | $series : no T1*.nii.gz produced"
      rm -rf "$tmp"
      ((failed+=1))
      continue
    fi

    mv -f "$nii" "$dst/T1.nii.gz"
    [[ -n "${jsn:-}" ]] && mv -f "$jsn" "$dst/T1.json" || true
    rm -rf "$tmp"

    # minimal debug info: dims + pixdims if fslinfo exists
    if command -v fslinfo >/dev/null 2>&1; then
      dim="$(fslinfo "$dst/T1.nii.gz" | awk '$1=="dim1"{d1=$2} $1=="dim2"{d2=$2} $1=="dim3"{d3=$2} END{printf "%sx%sx%s",d1,d2,d3}')"
      pix="$(fslinfo "$dst/T1.nii.gz" | awk '$1=="pixdim1"{p1=$2} $1=="pixdim2"{p2=$2} $1=="pixdim3"{p3=$2} END{printf "%sx%sx%s",p1,p2,p3}')"
      echo "[OK]      $sess | $series -> $dst/T1.nii.gz  dim=$dim  pix=$pix"
    else
      echo "[OK]      $sess | $series -> $dst/T1.nii.gz"
    fi

    ((converted+=1))
  done
done

echo
echo "Done. converted=$converted failed=$failed missing_sessions=$missing multi_sessions=$multi"
echo "OUT_ROOT=$OUT_ROOT"

