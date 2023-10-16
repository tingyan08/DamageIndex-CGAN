#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=excl
#PBS -o ./log/out_even.txt				
#PBS -e ./log/err_even.txt				
#PBS -N even

cd ~/ConditionalGAN 				

source ~/.bashrc			
conda activate cgan	

module load cuda-11.7			

# python3 main.py --gan_type CGAN_plus --dataset damage-index --epoch 40000 --batch_size 8 --input_size 128 --lrG 2e-4 --lrD 2e-4 --discrete_column AR HR VR --continuous_column DI
python samples_one_exp.py
