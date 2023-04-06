#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/mrichte/jupyter_output/jupyter.log

# Send some noteworthy information to the output log
echo "--- ACTIVATE APPROPRIATE ENVIRONMENT BEFORE EXECUTING!!! ---"
echo
echo "Running on node:  $(hostname)"
echo "In directory:     $(pwd)"
echo "Starting on:      $(date)"
echo "SLURM_JOB_ID:     ${SLURM_JOB_ID}"
echo "Running training script in conda environment: ${CONDA_DEFAULT_ENV}"

jupyter lab --no-browser --port 5998 --ip $(hostname -f)

# After a successful start, the notebook prints the URL it's accessible under, which looks similar to
# http://<hostname>.ee.ethz.ch:5998/?token=5586e5faa082d5fe606efad0a0033ad0d6dd898fe0f5c7af

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"
