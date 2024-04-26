#!/bin/sh
#SBATCH --gpus=1
#SBATCH

module purge
module load compilers/gcc/12.2.0 compilers/cuda/11.7 cudnn/8.6.0.163_cuda11.x anaconda/2021.11
source activate torch13
module unload anaconda/2021.11

#python train_davis_model1.py
python3 -m torch.distribued.launch --nproc_per_node=2 test_dist.py
