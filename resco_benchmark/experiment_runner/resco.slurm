#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --partition=xlong        #Set the partition to be used, fewer nodes for long but > 1 day jobs
#SBATCH --export=NONE                #Do not propagate environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=RESCO
#SBATCH --time=20:00:00               #Set the wall clock limit to hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --output=RESCO.%j

#SBATCH --account=132824387223

#First Executable Line
ml Miniconda3/23.5.2-0
source activate $SCRATCH/conda/resco/
cd $SCRATCH/RESCO/resco_benchmark/
echo "Use shell script"