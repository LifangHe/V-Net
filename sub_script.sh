#!/bin/bash

#SBATCH --job-name=mdgru                   #This is the name of your job
#SBATCH --cpus-per-task=4                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#Total memory reserved: 32GB
#SBATCH --partition=pascal     # or pascal / titanx
#SBATCH --gres=gpu:6        # --gres=gpu:2 for two GPU, aso.

#SBATCH --time=24:00:00        #This is the time that your task will run
#SBATCH --qos=1day           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=stdout     #This is the joined STDOUT and STDERR file
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=georgi.tancev@unibas.ch        #You will be notified via email when your task ends or fails

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $JOB_ID stores the ID number of your task.


#load your required modules below
#################################
ml Python/3.5.2-goolf-1.7.20
ml CUDA/9.0.176
ml cuDNN/7.3.1.20-CUDA-9.0.176


#export your required environment variables below
#################################################
source "/scicore/home/scicore/rumoke43/anaconda3/bin/activate" MIAC

#add your command lines below
#############################
python3 Training.py
