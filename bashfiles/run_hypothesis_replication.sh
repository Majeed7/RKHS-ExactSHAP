#!/bin/bash

# SLURM directives (optional, for cluster usage)
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

# Change directory to the home folder
cd ~/RKHS-ExactSHAP

# Load CUDA 12.1
module unload all
module load cuda12.3/toolkit/12.3

# Activate the Python environment
source /var/scratch/mmi454/envs/exactSV-PKM/bin/activate

# Print system info
echo "Python version:"
python --version
echo "CUDA version:"
nvcc --version

# Accept parameter from sbatch command
parameter=$1  # $1 refers to the first argument passed to the script

# Check if a parameter is passed
if [ -z "$parameter" ]; then
    echo "No argument passed. Using default value: 0.1"
    parameter=0  # Default value
fi

# Run the Python script
echo "Running Python script..."
python ~/RKHS-ExactSHAP/hypothesis_testing_replication.py "$parameter"


# Deactivate the environment
deactivate

