import Agents
from sweeper import Sweeper

from training_engine import TrainingEngine, ConfigDictConverter
from logger import OfflinePolicyEvaluationLogger

### offline_run
i_run = 0
sweeper = Sweeper('config/config.json')

# do some training here
config_dict = sweeper.parse(i_run)
cfg = ConfigDictConverter(config_dict)
# print(config_dict['run'])

# quit()
# print(config_dict)
# logger = MapGridWorldEpisodesLogger()  # pass the right things to the logger
# logger = SwitchGridWorldEpisodesLogger()
logger = cfg.logger_class(save_freq=100, save_model_freq=100, save_steps=True, save_episodes=False)

if config_dict['algorithm'][0:2] == "TD":
    save_tag = 'td'
elif config_dict['algorithm'][0:2] == "MC":
    save_tag = 'mc'
training_engine = TrainingEngine(cfg.env_class, cfg.config_dict, cfg.agent_class, cfg.config_dict, logger,
                                 config_dict["discount"], config_dict["num_steps"], i_run, config_dict['run'],
                                 iteration_counter="steps",
                                 save_file='test_save/' + save_tag,
                                 max_steps=config_dict["max_steps_per_ep"],
                                 offline_training=config_dict['offline_training']
                                 )
# num_runs=1 because repeats are taken care of using i_run.
# num_runs can be used to debug on single process

training_engine.run()