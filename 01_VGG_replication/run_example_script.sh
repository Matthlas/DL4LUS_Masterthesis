#!/bin/bash

echo "TEST SCRIPT RUNNING"
echo
echo "Running on node:  $(hostname)"
echo "In directory:     $(pwd)"
echo "Starting on:      $(date)"
echo "Running training script in conda environment: ${CONDA_DEFAULT_ENV}"

for i in 0 1 2 3 4
do
    echo "------------- FOLD $i -------------"
    python train_simple.py \
        --dataset covid_us \
        --model_dir PATH_TO_/models \
        --model_name test_covid_us_replicated \
        --fold $i \
        --epochs 2 \
        --stride 50
    echo "------------- FOLD $i FINISHED -------------"
    echo
    echo
done

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"