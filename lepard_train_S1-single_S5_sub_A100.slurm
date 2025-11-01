#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:1,gpumem_per_gpu:80GB
#SBATCH --time=80:00:00
#SBATCH --mem=512gb

module load devel/cuda/11.6
module load devel/miniforge/24.9.2
mamba activate lepard
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6'
#export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py configs/train/plants_S1-single_S5_sub.yaml
#python main.py configs/test/plants_S1-single_S5_sub.yaml
