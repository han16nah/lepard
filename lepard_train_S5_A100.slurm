#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:1,gpumem_per_gpu:80GB
#SBATCH --time=48:00:00
#SBATCH --mem=128gb

module load devel/cuda/11.6
module load devel/miniforge/24.9.2
mamba activate lepard
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6'
# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
#export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py configs/train/plants_S5.yaml
# python main.py configs/test/plants_S5.yaml
# run this separately and adapt path to S5_corrected
