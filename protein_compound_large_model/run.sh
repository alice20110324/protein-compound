#!/bin/bash
#SBATCH --gpus=1
#SBATCH -w paraai-n32-h-01-agent-97
module purge
module load cudnn/8.6.0.163_cuda11.x anaconda/2021.11 compilers/gcc/9.3.0 cuda/11.7.0 
source activate tor131cu117
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
python ddp_train_davis_model1.py --gpu 0,1,2
