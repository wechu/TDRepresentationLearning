#!/bin/bash
#SBATCH --account=def-dpmeger
#SBATCH --output=output/log_%A_%a.txt
#SBATCH --error=error/log_%A_%a.txt
#SBATCH --mail-user=chungwes@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --time=2:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb

module load python/3.8
module load scipy-stack
source py38/bin/activate

export OMP_NUM_THREADS=1

python data_generation/mountaincar_data.py test