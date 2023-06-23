#!/usr/bin/env bash

#SBATCH --array=0-192
#SBATCH --time=5:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32

srun python Run.py $SLURM_ARRAY_TASK_ID
