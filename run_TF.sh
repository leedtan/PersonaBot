#!/bin/bash
#SBATCH --job-name=c_dtf_personaef
#SBATCH --output=tf_persona.txt
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --constraint=gpu_12gb

module purge
module load python-3.5 cuda-8.0

python3 -u model.py --modelname tf_persona --server 1
