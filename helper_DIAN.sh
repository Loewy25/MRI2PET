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

[[ -d "$IN_ROOT" ]] || { echo "[FATAL] missing dir: $IN_ROOT"; exit 2; }

echo -e "sess_id\tseries\tnii_link\ttarget" > "$OUT_LIST"

# -L follows symlinks; -type f means "real file after following"
while IFS= read -r nii_link; do
  [[ -n "$nii_link" ]] || continue
  series="$(basename "$(dirname "$nii_link")")"
  sess_id="$(basename "$(dirname "$(dirname "$nii_link")")")"
  target="$(readlink -f "$nii_link" || true)"
  echo -e "${sess_id}\t${series}\t${nii_link}\t${target}" >> "$OUT_LIST"
done < <(find -L "$IN_ROOT" -type f -name "T1.nii.gz" | sort)

n_total=$(($(wc -l < "$OUT_LIST") - 1))
echo "[OK] total_candidates=$n_total"
echo "[INFO] sample:"
head -n 6 "$OUT_LIST"
