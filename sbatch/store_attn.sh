#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 5:00:00
#SBATCH --gpus=h100-80:8

#type 'man sbatch' for more information and options
#this job will ask for 1 full v100-32 GPU node(8 V100 GPUs) for 5 hours
#this job would potentially charge 40 GPU SUs

#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - PSC-username should be replaced by your PSC username
# - path-to-directory should be replaced by the path to your directory where the executable is
source ~/.bashrc
conda init
conda activate diffuse
cd /jet/home/billyli/mmmu_NoOps/mmmu
python extract_attention_scores.py --output_dir="sbatch_results_attn_fixed" --noop=all
#run pre-compiled program which is already in your project space

./gpua.out
