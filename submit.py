####
# Use this file to submit a job
# Uses the config file to specify the experiment
#
####
# Parts taken from AlphaEx by Yi Wan and Daniel Plop
#
####
import os
import time
from datetime import datetime
import shutil
from sweeper import *
# Can restrict the variable that are run in the sweep
# to any configuration of variables that match

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True, help="Name of config file")
parser.add_argument('--job', type=str, default="",
                    help="String denoting indexes to run (e.g 0-3,5). By default, runs all of them.")
# Note you can add %m to job_array_string to limit the number of simultaneous jobs to m

# parser.add_argument('--save_path', type=str, default='results/', help="Path to save results")
args = parser.parse_args()
# get arg for the save path and config file

# args.config_name = "example_config.json"
config_path = "config/"
sweep_config = config_path + args.config_name
sweeper = Sweeper(sweep_config)
save_id = datetime.today().strftime('%Y%m%d') + "_" + str(int(time.time()))[-6:]  # use date and Unix time as a unique ID


''' Submits a job to Slurm
sbatch_params: dict with variables for sbatch to use e.g. time, mem, cpu
export_params: dict with variables to export to the Slurm script
slurm_script_path: path to the .sh file for launching Slurm job (including the file name)
'''

# Arguments for Slurm
# sbatch_params = {}
sbatch_params = {"mem": "2gb",     # memory per node
                 "time": "2:59:00",  # HH:MM:SS
                 "cpus-per-task": "1"
                 }
export_params = {"CONFIG_NAME": args.config_name,
                 "SAVE_ID": save_id}
slurm_script_path = "submit_CC.sh"

# TODO
# download code from github
# ssh into server automatically and submit
# (well, this could be slightly dangerous and admit more bugs)
# Merge or zip files after the runs are done
# Could use project_root_dir so we can run submit from the folder above the folder containing the project
# then I'd have to change the sbatch params to be in another file. Basically, keep this file as a submitter
# but have separate submit files for each project. May cause confusion though.
# Might be easier to have separate folders for each project and no shared files.


# Make a directory to save files
os.makedirs(f"results/{save_id}", exist_ok=True)

# Save a copy of the config
shutil.copy2(sweep_config, f"results/{save_id}")

# Run sbatch file
arg_export = ",".join([f"{k}={v}" for k, v in export_params.items()])
arg_opt_sbatch = " ".join([f"--{k}={v}" for k, v in sbatch_params.items()])

if args.job == "":
    # default is to run all the jobs, divided into groups
    num_groups_of_runs = sweeper.get_num_groups_of_runs()
    args.job = f"0-{num_groups_of_runs - 1}"

bash_script = (
    f"sbatch "
    f"--array={args.job} "
    f"{arg_opt_sbatch} "
    f"--export={arg_export} "
    f"{slurm_script_path}"
)

# remove multiple spaces
bash_script = " ".join(bash_script.split())

print(bash_script)
myCmd = os.popen(bash_script).read()
print(myCmd)
print(f"job array {args.job}, {sbatch_params}")


#### Can add another job that starts only after the previous array job is done
# use --depend on sbatch
# Can use this to do postprocessing on all the files afterwards
# e.g. sbatch --depend=afterok{slurm_job_id} postprocess_CC.sh
# Use this to zip a bunch of result files together or merge the results in single arrays
# Could also produce plots and summaries automatically after the job is done

