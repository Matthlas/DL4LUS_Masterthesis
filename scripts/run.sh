#!/bin/bash

python create_image_dataset.py 
#sbatch --gres=gpu:1 run.sh