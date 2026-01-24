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

IN_ROOT="/ceph/chpc/mapped/dian_obs_data_shared/obs_mr_scans_imagids"
OUT_ROOT="/scratch/l.peiwang/DIAN"
mkdir -p "$OUT_ROOT"

INC_RE='mprage|mp[-_ ]?rage|spgr|fspgr|ir[-_ ]?fspgr|bravo|tfe'
EXC_RE='moco|mosaic|localizer|3[_ -]?plane|phoenix|zipreport|report|field|mag|pha|phase|swi|mip|flair|t2|dti|diff|adc|tracew|tensor|colfa|fa|fmri|rsfmri|rest|asl|perfusion|default_ps_series|surv_new_scale_parameters|unknown'

sanitize() { echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

missing=0; multi=0; converted=0; failed=0

for sess_dir in "$IN_ROOT"/*_mr; do
  [[ -d "$sess_dir" ]] || continue
  sess="$(basename "$sess_dir")"
  out_sess="$OUT_ROOT/$sess"
  mkdir -p "$out_sess"

  cand_list="$(find "$sess_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
    | egrep -i "$INC_RE" | egrep -iv "$EXC_RE" | sort || true)"

  if [[ -z "$cand_list" ]]; then
    echo "[MISSING] $sess : no T1 candidates"
    ((missing+=1))
    continue
  fi

  cand_count="$(printf "%s\n" "$cand_list" | wc -l | awk '{print $1}')"
  if [[ "$cand_count" -gt 1 ]]; then
    echo "[MULTI]   $sess : $cand_count candidates"
    ((multi+=1))
  else
    echo "[ONE]     $sess : 1 candidate"
  fi

  while IFS= read -r series; do
    [[ -n "$series" ]] || continue
    src="$sess_dir/$series"

    # handle nested DICOM folder
    dicom_dir="$src"
    [[ -d "$src/DICOM" ]] && dicom_dir="$src/DICOM"

    series_safe="$(sanitize "$series")"
    dst="$out_sess/$series_safe"
    mkdir -p "$dst"

    tmp="$dst/.tmp_dcm2niix"
    rm -rf "$tmp"; mkdir -p "$tmp"

    if ! dcm2niix -z y -f T1 -o "$tmp" "$dicom_dir" >/dev/null 2>&1; then
      echo "[FAIL]    $sess | $series : dcm2niix error (dir=$(basename "$dicom_dir"))"
      rm -rf "$tmp"
      ((failed+=1))
      continue
    fi

    nii="$(ls -1t "$tmp"/T1*.nii.gz 2>/dev/null | head -n 1 || true)"
    jsn="$(ls -1t "$tmp"/T1*.json 2>/dev/null | head -n 1 || true)"

    if [[ -z "${nii:-}" ]]; then
      echo "[FAIL]    $sess | $series : no T1*.nii.gz produced (dir=$(basename "$dicom_dir"))"
      rm -rf "$tmp"
      ((failed+=1))
      continue
    fi

    mv -f "$nii" "$dst/T1.nii.gz"
    [[ -n "${jsn:-}" ]] && mv -f "$jsn" "$dst/T1.json" || true
    rm -rf "$tmp"

    if command -v fslinfo >/dev/null 2>&1; then
      dim="$(fslinfo "$dst/T1.nii.gz" | awk '$1=="dim1"{d1=$2} $1=="dim2"{d2=$2} $1=="dim3"{d3=$2} END{printf "%sx%sx%s",d1,d2,d3}')"
      pix="$(fslinfo "$dst/T1.nii.gz" | awk '$1=="pixdim1"{p1=$2} $1=="pixdim2"{p2=$2} $1=="pixdim3"{p3=$2} END{printf "%sx%sx%s",p1,p2,p3}')"
      echo "[OK]      $sess | $series -> $dst/T1.nii.gz  dim=$dim  pix=$pix"
    else
      echo "[OK]      $sess | $series -> $dst/T1.nii.gz"
    fi

    ((converted+=1))
  done <<< "$cand_list"
done

echo
echo "Done. converted=$converted failed=$failed missing_sessions=$missing multi_sessions=$multi"
echo "OUT_ROOT=$OUT_ROOT"
