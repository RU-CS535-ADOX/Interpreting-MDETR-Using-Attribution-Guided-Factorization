#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/xl598/Desktop/mdetr/checkpoint/xl598/experiments/%j/%j_0_log.err
#SBATCH --gpus-per-node=4
#SBATCH --job-name=detectransformer
#SBATCH --mail-type=END
#SBATCH --mail-user=helloimlixin@gmail.com
#SBATCH --mem=4GB
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/home/xl598/Desktop/mdetr/checkpoint/xl598/experiments/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR1@120
#SBATCH --time=4300
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /home/xl598/Desktop/mdetr/checkpoint/xl598/experiments/%j/%j_%t_log.out --error /home/xl598/Desktop/mdetr/checkpoint/xl598/experiments/%j/%j_%t_log.err --unbuffered /home/xl598/.conda/envs/mdetr_env/bin/python -u -m submitit.core._submit /home/xl598/Desktop/mdetr/checkpoint/xl598/experiments/%j
