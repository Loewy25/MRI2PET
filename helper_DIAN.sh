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

IN_ROOT="/scratch/l.peiwang/DIAN_geom"
OUT_LIST="/scratch/l.peiwang/dian_geom_all_t1_list.tsv"

echo "[INFO] IN_ROOT=$IN_ROOT"
echo "[INFO] OUT_LIST=$OUT_LIST"

echo -e "sess_id\tseries\tnii" > "$OUT_LIST"

# Expect layout: IN_ROOT/<sess_id>/<series>/T1.nii.gz
find "$IN_ROOT" -type f -name "T1.nii.gz" | sort | while read -r nii; do
  series="$(basename "$(dirname "$nii")")"
  sess_id="$(basename "$(dirname "$(dirname "$nii")")")"
  echo -e "${sess_id}\t${series}\t${nii}" >> "$OUT_LIST"
done

n_total=$(($(wc -l < "$OUT_LIST") - 1))
echo "[OK] wrote $OUT_LIST"
echo "[OK] total_candidates=$n_total"
echo "[INFO] sample:"
head -n 5 "$OUT_LIST"
