#!/bin/bash
#SBATCH --job-name=TE_Sliding_Marmoset_perm_in_time_new_electrodes
#SBATCH --output=out_perm_%j.log
#SBATCH --partition=64GBLppc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem=50G                 


# Load modules and environment
module load conda
conda activate idtxl

# Run the python script
python -u Script2_TE_on_cluster.py
