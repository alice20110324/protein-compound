#!/bin/bash
#SBATCH --gpus=1
#SBATCH -w paraai-n32-h-01-agent-97
module purge
module load compilers/gcc/12.2.0 compilers/cuda/11.7 cudnn/8.6.0.163_cuda11.x anaconda/2021.11
source activate torch13
module unload anaconda/2021.11

#python train_davis_model111.py
python ddp_train_davis_model1_dataset.py --gpu 0,1,2,3,4,5,6
