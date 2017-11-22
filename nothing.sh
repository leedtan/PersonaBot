#!/bin/bash
#SBATCH --job-name=chat_nothing
#SBATCH --output=chat_nothing.txt
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --constraint=gpu_12gb

module purge
module load python-3.5 cuda-8.0

USE_CUDA=1 python3 -u model.py --modelname chatbot_nothing --server 1 --adversarial_sample 0 --lambda_repetitive .02 --lambda_reconstruct .01
