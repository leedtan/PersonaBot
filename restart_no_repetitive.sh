#!/bin/bash
#SBATCH --job-name=chat_no_repetitive
#SBATCH --output=chat_no_repetitive.txt
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --constraint=gpu_12gb

module purge
module load python-3.5 cuda-8.0

USE_CUDA=1 python3 -u model.py --modelnameload chatbot_no_repetitive --server 1 --lambda_repetitive .1 --loaditerations 120000
