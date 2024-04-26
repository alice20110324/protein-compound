#!/bin/bash
#SBATCH --gpus=1
module purge
module load compilers/gcc/12.2.0 compilers/cuda/11.7 cudnn/8.6.0.163_cuda11.x anaconda/2021.11
source activate torch13
module unload anaconda/2021.11

#export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=4 minist_dist_test.py
