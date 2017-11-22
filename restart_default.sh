#!/bin/bash
#SBATCH --job-name=chat_default
#SBATCH --output=chat_default.txt
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --constraint=gpu_12gb

module purge
module load python-3.5 cuda-8.0

USE_CUDA=1 python3 -u model.py --modelnameload chatbot_default --server 1 --loaditerations 110000
