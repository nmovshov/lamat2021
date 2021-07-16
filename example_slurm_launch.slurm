#!/bin/bash
#SBATCH --job-name=run1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nmovshov@ucsc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --time=64:00:00

. /home/nmovshov/miniconda3/etc/profile.d/conda.sh
conda activate

python driver1.py req1 req2 -opt1=1 --optional2=2 >& rlog.$SLURM_JOB_ID.log

python driver2.py req1 >> rlog.$SLURM_JOB_ID.log
