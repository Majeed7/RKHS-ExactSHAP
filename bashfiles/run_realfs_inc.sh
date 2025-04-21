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

# Run the Python script
echo "Running Python script..."
for param in {1..30}; do
    echo "Running with parameter: $param"
    python ~/RKHS-ExactSHAP/incremental_addingfeatures.py --input "$param"
done


# Deactivate the environment
deactivate

