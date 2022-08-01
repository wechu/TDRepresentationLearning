#####
# Classes for logging data from runs
#
#
#####
import abc
from collections import defaultdict

import numpy as np
import pickle


class BaseMetricLogger(abc.ABC):
    def __init__(self, save_freq, save_episodes, save_steps):
        self.metrics = defaultdict(list)
        self.run_metrics = []
        self.save_freq = save_freq
        self.save_episodes = True  # use if we want to save at the end of episodes
        self.save_steps = True  # use if we want to save after each step

    def save_episode_metrics(
        self,
        num_steps,
        trajectory,
        total_reward,
        total_discounted_reward,
        agent,
        env,
        episode_end_step,
        *args,
        **kwargs
    ):
        self.metrics["returns"].append(total_reward)
        self.metrics["discounted_returns"].append(total_discounted_reward)
        self.metrics["episodes_end_step"].append(episode_end_step)  # how many steps elapsed before each episode ended

    def save_step_metrics(self, state, action, reward, done, *args, **kwargs):
        self.metrics['states'].append(state)
        self.metrics['actions'].append(action)
        self.metrics['rewards'].append(reward)
        self.metrics['dones'].append(done)

    # def save_evaluation_run_metrics(
    #     self,
    #     num_steps,
    #     trajectory,
    #     total_reward,
    #     total_discounted_reward,
    #     agent,
    #     env,
    #     *args,
    #     **kwargs
    # ):
    #     pass

    def reset(self):
        self.metrics = defaultdict(list)

    def record_run(self):
        self.run_metrics.append(self.metrics)

    def save_to_file(self, save_file):
        with open(f'{save_file}.pkl', 'wb') as file:
            pickle.dump(self.metrics, file, protocol=pickle.HIGHEST_PROTOCOL)


class AllStepsLogger(BaseMetricLogger):
    def __init__(self):
        '''
        This logger saves every transition (state, action, reward) the agent experiences at every step
        '''
        super().__init__(save_freq=1, save_episodes=True, save_steps=True)

    def save_step_metrics(self, state, action, reward, done, *args, **kwargs):

        super().save_step_metrics(state, action, reward, done)  # saves these four quantities

        # Note: In partially-observable envs, this only saves the obs, not the true state of the environment
        # Can pass the environment to this function to also save the env state

    def save_to_file(self, save_file):
        # save in more efficient data types (for gridworld)
        save_dict = dict()
        save_dict['states'] = np.array(self.metrics['states'], dtype='int8')
        save_dict['actions'] = np.array(self.metrics['actions'], dtype='int8')
        save_dict['rewards'] = np.array(self.metrics['rewards'], dtype='float32')
        save_dict['dones'] = np.array(self.metrics['dones'], dtype='bool')

        save_dict['returns'] = np.array(self.metrics['returns'], dtype='float32')
        save_dict['discounted_returns'] = np.array(self.metrics['discounted_returns'], dtype='float32')
        save_dict['episodes_end_step'] = np.array(self.metrics["episodes_end_step"], dtype='int32')

        with open(f'{save_file}.pkl', 'wb') as file:
            pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)



class OfflinePolicyEvaluationLogger:
    def __init__(self, save_freq, save_episodes, save_steps, save_model_freq=None):
        self.metrics = defaultdict(list)
        self.run_metrics = []
        self.save_freq = save_freq
        self.save_episodes = False  # use if we want to save at the end of episodes
        self.save_steps = True  # use if we want to save after each step
        self.save_model_freq = save_model_freq  # how often to save the model checkpoints (less frequent usually)

    def save_step_metrics(self, option, agent=None, total_num_steps=None, test_error=None, save_file=None, *args, **kwargs):
        # save model
        if option == 'standard':
            self.metrics['test_error'].append(test_error)
        elif option == 'model':
            agent.save_model(save_file + f'_model_{total_num_steps}.pyt')

    def reset(self):
        self.metrics = defaultdict(list)

    def record_run(self):
        self.run_metrics.append(self.metrics)

    def save_to_file(self, save_file):
        with open(f'{save_file}.pkl', 'wb') as file:
            pickle.dump(self.metrics, file, protocol=pickle.HIGHEST_PROTOCOL)


class MapGridWorldEpisodesLogger(BaseMetricLogger):
    def __init__(self):
        '''
        This logger saves statistics at the end of every episode
        It also keeps summaries of the visitation
        Designed for the map gridworld
        '''
        self.env_name = "MapGridWorld"
        super().__init__(save_freq=1, save_episodes=True, save_steps=False)

    def save_episode_metrics(
        self,
        num_steps,
        trajectory,
        total_reward,
        total_discounted_reward,
        agent,
        env,
        episode_end_step,
        *args,
        **kwargs
    ):
        # trajectory is a list of tuples (state, action, reward, done)
        self.metrics['returns'].append(total_reward)
        self.metrics['discounted_returns'].append(total_discounted_reward)
        self.metrics['num_steps'].append(num_steps)
        self.metrics['picked_up_map'].append(env.picked_up_map)
        self.metrics['time'].append(kwargs['time'])
        # state-visitation
        # if env.name == 'gridworld':
        #     state_visitation = np.zeros(env.gridsize, dtype='uint8')
        #     for s, a, r, done in trajectory:
        #         state_visitation[s[0], s[1]] += 1
        #     self.metrics['state_visitation'].append(state_visitation)

    def save_to_file(self, save_file):
        # save in more efficient data types (for gridworld)
        save_dict = dict()

        save_dict['returns'] = np.array(self.metrics['returns'], dtype='float32')
        save_dict['discounted_returns'] = np.array(self.metrics['discounted_returns'], dtype='float32')
        save_dict['num_steps'] = np.array(self.metrics['num_steps'], dtype='int32')

        save_dict['picked_up_map'] = np.array(self.metrics['picked_up_map'])
        # save_dict['state_visitation'] = np.stack(self.metrics['state_visitation'])
        save_dict['time'] = np.array(self.metrics['time'])
        with open(f'{save_file}.pkl', 'wb') as file:
            pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


