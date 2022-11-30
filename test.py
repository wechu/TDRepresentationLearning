import Agents
from sweeper import Sweeper

from training_engine import TrainingEngine, ConfigDictConverter
from logger import OfflinePolicyEvaluationLogger

### offline_run
# i_runs = [0,1,2,3]
i_runs = [0]

for i_run in i_runs:
    sweeper = Sweeper('config/config_test.json')

    # do some training here
    config_dict = sweeper.parse(i_run)
    cfg = ConfigDictConverter(config_dict)
    # print(config_dict['run'])

    # quit()
    # print(config_dict)
    # logger = MapGridWorldEpisodesLogger()  # pass the right things to the logger
    # logger = SwitchGridWorldEpisodesLogger()
    logger = cfg.logger_class(save_freq=config_dict['save_freq'], save_model_freq=config_dict['save_model_freq'], save_steps=True, save_episodes=False)

    if config_dict['algorithm'][0:2] == "TD":
        save_tag = 'td_early'
    elif config_dict['algorithm'][0:2] == "MC":
        save_tag = 'mc_early'
    training_engine = TrainingEngine(cfg.env_class, cfg.config_dict, cfg.agent_class, cfg.config_dict, logger,
                                     config_dict["num_steps"], 0, config_dict['run'],
                                     iteration_counter="steps",
                                     save_file=f'test_save_{config_dict["env"].lower()}/' + save_tag,
                                     max_steps=config_dict["max_steps_per_ep"],
                                     offline_training=config_dict['offline_training']
                                     )

    # num_runs=1 because repeats are taken care of using i_run.
    # num_runs can be used to debug on single process

    training_engine.run()