#!/bin/bash
#SBATCH --job-name=fehydro
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=10:00:00

# Optional modules / conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lno_pyscf_afqmc

# Run with srun
srun python test.py |tee ueg_33.out
