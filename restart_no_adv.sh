#!/bin/bash
#SBATCH --job-name=chat_no_adv
#SBATCH --output=chat_no_adv.txt
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --constraint=gpu_12gb

module purge
module load python-3.5 cuda-8.0

USE_CUDA=1 python3 -u model.py --modelnameload chatbot_no_adv --server 1 --adversarial_sample 0 --loaditerations 160000
