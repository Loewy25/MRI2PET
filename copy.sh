#!/bin/bash
#SBATCH --job-name=copy_aris_project
#SBATCH --mem=10G
#SBATCH --time=23:50:00
#SBATCH --partition=tier1_gpu
#SBATCH --account=shinjini_kundu
#SBATCH --output=slurm-%A_copy.out
#SBATCH --error=slurm-%A_copy.err

# -----------------------------
# Paths
# -----------------------------

rm -rf /scratch/l.peiwang/aris_project
rm -rf /scratch/l.peiwang/l.peiwang_scratch
SRC="/ceph/chpc/shared/aristeidis_sotiras_group/l.peiwang_scratch"
DST="/scratch/l.peiwang/aris_project"

echo "Starting copy job at $(date)"
echo "Source:      $SRC"
echo "Destination: $DST"
echo "Running on node: $(hostname)"

# -----------------------------
# Safety checks
# -----------------------------
if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory does not exist"
    exit 1
fi

mkdir -p "$DST"

# -----------------------------
# Copy (rsync is safer than cp)
# -----------------------------
rsync -avh --progress "$SRC/" "$DST/"

echo "Copy finished at $(date)"
