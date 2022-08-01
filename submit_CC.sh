#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --output=output/log_%A_%a.txt
#SBATCH --error=error/log_%A_%a.txt
#SBATCH --mail-user=chungwes@mila.quebec
#SBATCH --mail-type=ALL

module load python/3.8
module load scipy-stack
source py38/bin/activate

export OMP_NUM_THREADS=1

python run.py --sbatch_idx $SLURM_ARRAY_TASK_ID --config_name $CONFIG_NAME --save_id $SAVE_ID
