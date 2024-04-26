#!/bin/bash
#SBATCH -w node209
#SBATCH --gres=gpu:1
/public/software/anaconda3/envs/pytorch/bin/python3 train_refined_set.py

