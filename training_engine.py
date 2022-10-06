import time
from inspect import signature
import os, psutil

import gym.envs.classic_control
import numpy as np

import deepRL
import Agents
import offline_data


class TrainingEngine:
    # Defines the name of the environment argument in the agent class
    # if the agent class is to be initialized with the env
    AGENT_ENV_ARG = "env"

    def __init__(
            self,
            env_class,
            env_parameters,
            agent_class,
            agent_parameters,
            metric_logger,
            num_iterations_per_run,
            config_index=-1,
            repeat_index=-1,
            num_runs=1,
            iteration_counter="episodes",
            evaluation_num_episodes=None,
            pass_env_to_agent=True,
            max_steps=None,
            save_file=None,
            offline_training=False):
        # If you change the arguments, you have to check run.py
        self.env_class = env_class
        self.env_parameters = env_parameters
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.metric_logger = metric_logger
        self.num_runs = num_runs  # number of repeats to do within the loop.
        self.num_iterations_per_run = num_iterations_per_run
        self.config_index = config_index
        self.repeat_index = repeat_index
        self.iteration_counter = iteration_counter  # episodes or steps
        self.evaluation_num_episodes = evaluation_num_episodes
        self.pass_env_to_agent = pass_env_to_agent
        self.max_steps = max_steps
        self.offline_training = offline_training

        self.discount_rate = self.agent_parameters['discount']


        if save_file is None:
            self.save_file = f"res_{config_index}"
        else:
            self.save_file = save_file + f"_{config_index}" # string
        self._sanity_check()

    def run(self):
        if self.offline_training:
            self.offline_run()
        else:
            self.online_run()

    def online_run(self):
        print("Start!")
        start_time = time.perf_counter()

        for _ in range(self.num_runs):
            self.metric_logger.reset()  # reset for run
            env, agent = self._init_env_agent()

            total_num_steps = [0]
            i_ep = [0]
            counter = i_ep if self.iteration_counter == "episodes" else total_num_steps  # uses mutability of lists
            break_value = self.num_iterations_per_run

            while counter[0] < break_value:
                state = env.reset()
                done = False
                trajectory = []
                total_reward = 0
                total_discounted_reward = 0
                num_steps = 0

                while not done:
                    prev_state = state
                    # print(state)
                    action = agent.act(state)
                    state, reward, done, info = env.step(action)

                    total_reward += reward
                    total_discounted_reward += reward * self.discount_rate ** num_steps

                    trajectory.append((prev_state, action, reward, done))

                    # print("tr", action, reward, state)
                    loss = agent.online_update(prev_state, action, reward, state, done)
                    # if loss is not None:
                    #     print(loss)

                    if self.metric_logger.save_steps and (total_num_steps[0] == 0 or (total_num_steps[0] + 1) % self.metric_logger.save_freq == 0):
                        self.metric_logger.save_step_metrics(
                            state=prev_state, action=action, reward=reward, done=done
                        )

                    num_steps += 1
                    total_num_steps[0] += 1
                    if (self.max_steps and num_steps >= self.max_steps) or (counter[0] >= break_value):
                        break

                agent.episode_update(trajectory=trajectory, num_steps=num_steps)

                extra_log = {}

                if self.metric_logger.save_episodes and (i_ep[0] == 0 or (i_ep[0] + 1) % self.metric_logger.save_freq == 0):
                    self.metric_logger.save_episode_metrics(
                        num_steps=num_steps,
                        trajectory=trajectory,
                        total_reward=total_reward,
                        total_discounted_reward=total_discounted_reward,
                        agent=agent,
                        env=env,
                        episode_end_step=total_num_steps,
                        time=round((time.perf_counter() - start_time) / 60,3),
                        **extra_log
                    )

                i_ep[0] += 1

            self.metric_logger.record_run()

        # self.metric_logger.save(agent_param_id=self.agent_param_id,
        #                         agent_parameters=self.agent_parameters)
        self.metric_logger.save_to_file(self.save_file)
        return self.metric_logger

    def offline_run(self):
        ''' Runs using offline datasets '''
        # load data
        # agent.initialize_data()
        # for i_step in range(num_steps):
        #   agent.update(transition)
        #   logger.save(stuff)

        print(f"Start offline run! Index {self.config_index}")
        start_time = time.perf_counter()

        for _ in range(self.num_runs):
            self.metric_logger.reset()  # reset for run
            env, agent = self._init_env_agent()

            data = offline_data.load_train_data(self.agent_parameters['offline_data_path'],
                                                self.agent_parameters['env'],
                                                self.repeat_index  # todo does this work? we need the hyperparameter index
                                                )
            agent.initialize_data(data)

            total_num_steps = 0  # episodes don't make sense in offline RL

            while total_num_steps < self.num_iterations_per_run:
                loss = agent.offline_update()
                extra_log = {}

                # if total_num_steps % 100 == 0:
                #     print(total_num_steps)
                #     print("test error", self._test_policy_evaluation(agent))

                if self.metric_logger.save_steps:

                    if total_num_steps == 0 or (total_num_steps + 1) % self.metric_logger.save_freq == 0:
                        # run external evaluation
                        test_error = self._test_policy_evaluation(agent)
                        print(total_num_steps, "test error", test_error)
                        self.metric_logger.save_step_metrics(option='standard',
                                                             agent=agent,
                                                             test_error=test_error,
                                                             loss=loss,
                                                             **extra_log)

                    if total_num_steps == 0 or (total_num_steps+1) % self.metric_logger.save_model_freq == 0:
                        self.metric_logger.save_step_metrics(option='model',
                                                             agent=agent,
                                                             total_num_steps=total_num_steps,
                                                             **extra_log,
                                                             save_file=self.save_file)
                total_num_steps += 1

            self.metric_logger.record_run()

        # self.metric_logger.save(agent_param_id=self.agent_param_id,
        #                         agent_parameters=self.agent_parameters)
        self.metric_logger.save_to_file(self.save_file)
        print('done offline run')
        return self.metric_logger

    def _init_env_agent(self):
        # To create envs with parameters, we need an environment constructor that produces the appropriate class
        # del self.env_parameters['run']
        if self.offline_training:
            env = None
        else:
            env = self.env_class(**self.env_parameters)  # TODO check how this works with gym envs

            if self.pass_env_to_agent:
                self.agent_parameters.update({self.AGENT_ENV_ARG: env})

        agent = self.agent_class(**self.agent_parameters)

        return env, agent

    def _sanity_check(self):
        if self.pass_env_to_agent:
            agent_sig = signature(self.agent_class)
            if not agent_sig.parameters.get(self.AGENT_ENV_ARG):
                raise ValueError(f"When pass_env_agent is set to True, \
                    agent_class needs to have a named argument {self.AGENT_ENV_ARG}")

        if self.iteration_counter not in ["episodes", "steps"]:
            raise ValueError(f'iteration_counter must be either "episodes" or "steps", \
                received {self.iteration_counter} instead')

    def _test_policy_evaluation(self, agent):
        test_states = offline_data.load_test_states(self.agent_parameters['offline_data_path'],
                                                    self.agent_parameters['env'])
        test_values = offline_data.load_test_values(self.agent_parameters['offline_data_path'],
                                                    self.agent_parameters['env'])

        return test_policy_evaluation_error(agent, test_states, test_values)


    def _external_evaluation(self, agent, evaluation_num_episodes):
        env, _ = self._init_env_agent()
        for _ in range(evaluation_num_episodes):
            state = env.reset()
            done = False
            trajectory = []
            total_reward = 0
            total_discounted_reward = 0
            num_steps = 0

            while not done:
                prev_state = state

                action = agent.get_action(state)
                state, reward, done = env.step(action)

                total_reward += reward
                total_discounted_reward += reward * (self.discount_rate ** num_steps)

                trajectory.append((prev_state, action, reward))
                if self.max_steps and num_steps >= self.max_steps:
                    break

            self.metric_logger.save_evaluation_run_metrics(
                num_steps=num_steps,
                trajectory=trajectory,
                total_reward=total_reward,
                total_discounted_reward=total_discounted_reward,
                agent=agent,
                env=env)


def test_policy_evaluation_error(agent, test_states, test_values):
    # computes the root-mean-squared-error (RMSE)
    errors = []
    for test_state, test_value in zip(test_states, test_values):
        pred = agent.get_prediction(test_state)
        errors.append((pred - test_value)**2)
    # print(errors)
    return np.sqrt(np.mean(errors))

import Envs
import Agents
import logger
class ConfigDictConverter:
    def __init__(self, config_dict):
        '''
        This class takes a config_dict which contains all the variables needed to do one run
        and converts it into the variables needed to run the RL experiments
        We assume that the config file has certain variables and is organized in the proper way
        For the env and agent parameters, we will pass config_dict on to them and assume that they handle it properly
        Note that we *cannot* use the same variable names for env and agent parameters. If the agent or env expect the
        same name, we will need to write it down different in the config file and then convert here
        '''
        self.config_dict = config_dict.copy()
        # training engine shouldn't need these variables
        del self.config_dict['num_repeats']
        del self.config_dict['num_runs_per_group']

        self.repeat_idx = config_dict['run']

        # Associate the environment class
        if config_dict['env'] == "GridWorld":
            self.env_class = Envs.GridWorldEnv
            self.logger_class = logger.MapGridWorldEpisodesLogger

        # algorithm
        # print(config_dict['algorithm'])
        if config_dict['algorithm'] == "GridWorldQlearning":
            self.agent_class = Agents.GridWorldQLearning
        if config_dict['algorithm'] == "QR-DQN":
            self.agent_class = deepRL.QRDQN_Agent
        if config_dict['algorithm'] == "TDPolicyEval":
            self.agent_class = Agents.OfflinePolicyEvalTDAgent
        if config_dict['algorithm'] == "MCPolicyEval":
            self.agent_class = Agents.OfflinePolicyEvalMCAgent

        # offline RL. Add data paths
        if config_dict['env'].lower() == 'cartpole' and config_dict['offline_training']:
            self.config_dict['offline_data_path'] = "data_generation/"
            self.env_class = gym.envs.classic_control.CartPoleEnv
            self.logger_class = logger.OfflinePolicyEvaluationLogger
            self.config_dict['state_size'] = 4
            self.config_dict['action_size'] = 3
        elif (('mountaincar' == config_dict['env'].lower() or 'sparse_mountaincar' == config_dict['env'].lower())
              and config_dict['offline_training']):
            self.config_dict['offline_data_path'] = "data_generation/"
            self.env_class = gym.envs.classic_control.MountainCarEnv
            self.logger_class = logger.OfflinePolicyEvaluationLogger
            self.config_dict['state_size'] = 2
            self.config_dict['action_size'] = 3
        elif config_dict['env'].lower() == 'wall_gridworld':
            self.config_dict['offline_data_path'] = "data_generation/"
            self.env_class = Envs.WallGridWorldEnv
            self.logger_class = logger.OfflinePolicyEvaluationLogger
            self.config_dict['state_size'] = 2
            self.config_dict['action_size'] = 5
        elif config_dict['env'].lower() == 'toy': # todo finish
            self.config_dict['offline_data_path'] = "data_generation/"
            self.config_dict['state_size'] = 2
            self.config_dict['action_size'] = 1
        else:
            raise AssertionError("config dict converter: env doesn't match")

        # TODO env parameters could be filtered to only contain what is needed here

        # # add other variables
        # # for agent
        # TODO change the state size or hardcode
        # self.config_dict['state_size'] = self.env_class(**config_dict).state_size
        # self.config_dict['action_size'] = self.env_class(**config_dict).num_actions







