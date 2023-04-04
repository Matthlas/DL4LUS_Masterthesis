#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-380

# For array job range find the number of .npy files in current directory with "ls *.npy | wc -l" and subtract 1
# 396 files in total

# Send some noteworthy information to the output log
echo "--- ACTIVATE TDA ENVIRONMENT BEFORE EXECUTING!!! ---"
echo
echo "Running on node:  $(hostname)"
echo "In directory:     $(pwd)"
echo "Starting on:      $(date)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID}"
echo "Array index:      ${SLURM_ARRAY_TASK_ID}"
echo "Running training script in conda environment: ${CONDA_DEFAULT_ENV}"


python calculate_cubicial_persistence_entire_video.py \
    --file_no ${SLURM_ARRAY_TASK_ID}

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"