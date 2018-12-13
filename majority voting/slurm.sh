#!/usr/bin/env bash
#SBATCH --job-name=RUnet_comb
#SBATCH --partition=slurm_shortgpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:40:0
#SBATCH --output="%j.txt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gtx1080:1

cd $SLURM_SUBMIT_DIR

module load python/3.6.0
module load groupmods/me539/cuda
# module load groupmods/me539/tensorflow
module list

python3 py_comb.py
