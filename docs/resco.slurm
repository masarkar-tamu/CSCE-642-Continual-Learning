#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --partition=long        #Set the partition to be used, fewer nodes for long but > 1 day jobs
#SBATCH --export=NONE                #Do not propagate environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=RESCO
#SBATCH --time=168:00:00               #Set the wall clock limit to hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=384G
#SBATCH --output=RESCO.%j

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=jault@tamu.edu    #Send all emails to email_address

#First Executable Line
jobstats &
ml load GCCcore/13.2.0 Python/3.11.5
source $SCRATCH/venvs/resco/bin/activate
export SUMO_HOME=$SCRATCH/venvs/resco/lib/python3.11/site-packages/sumo
export LIBSUMO_AS_TRACI=true
cd $SCRATCH/projects/priv_RESCO/resco_benchmark/
## Some issue with compression on the cluster
###sed -i 's%zip=True%zip=False%' main.py
## Manually input runner to start, set ntasks, nodes, and time
$SCRATCH/venvs/resco/bin/python3 runner_
tamulauncher _commands_
jobstats