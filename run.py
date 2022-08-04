#####
# sbatch script runs this file
#
# Gets args for the config file (or hardcode?)
# Reads the config file, calls sweeper, figures out which sweep_ids to run based on grouping,
# Gets the parameter dictionary and then runs the appropriate experiment based on this
#####
import argparse
from sweeper import Sweeper
from training_engine import TrainingEngine, ConfigDictConverter
from logger import OfflinePolicyEvaluationLogger

parser = argparse.ArgumentParser()
parser.add_argument('--sbatch_idx', type=int, required=False, help='Index passed by sbatch (SLURM_ARRAY_TASK_ID)')
parser.add_argument('--config_name', type=str, required=False, help='Name of the config file')
parser.add_argument('--save_id', type=str, required=False, help='ID associated to this sbatch call (save directory)')
args = parser.parse_args()

save_path = f"results/{args.save_id}/"

sweeper = Sweeper(save_path + args.config_name)
idx_to_run = sweeper.convert_sbatch_idx_into_sweep_idx(args.sbatch_idx)

for i_run in idx_to_run:
    # do some training here
    config_dict = sweeper.parse(i_run)
    cfg = ConfigDictConverter(config_dict)

    # print(config_dict)
    # logger = MapGridWorldEpisodesLogger()  # pass the right things to the logger
    # logger = SwitchGridWorldEpisodesLogger()
    logger = cfg.logger_class(save_freq=config_dict['save_freq'], save_model_freq=config_dict['save_model_freq'],
                              save_steps=True, save_episodes=False)

    training_engine = TrainingEngine(cfg.env_class, cfg.config_dict, cfg.agent_class, cfg.config_dict, logger,
                                     config_dict["num_steps"], i_run, config_dict['run'],
                                     iteration_counter="steps",
                                     save_file=save_path + 'res',
                                     max_steps=config_dict["max_steps_per_ep"],
                                     offline_training=config_dict['offline_training']
                                     )
    # num_runs=1 because repeats are taken care of using i_run.
    # num_runs can be used to debug on single process

    training_engine.run()
    print(f"Done run {i_run}")

