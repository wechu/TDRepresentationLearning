#######################################################################
# Copyright (C) 2019 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import json


class Sweeper(object):
    """
    The purpose of this class is to take an index, identify a configuration
    of variables and create a Config object
    Important: variables part of the sweep are provided in a list
    """

    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1  # this doesn't include repeats
        self.keys_list = []
        self.set_num_combinations()
        self.keys_set = set(self.keys_list)

        self.num_repeats = self.config_dict["num_repeats"][0]
        self.num_runs_per_group = self.config_dict["num_runs_per_group"][0] if "num_runs_per_group" in self.config_dict else 1

    def set_num_combinations(self):
        # calculating total_combinations
        self.set_num_combinations_helper(self.config_dict)
        self.total_combinations = self.config_dict["num_combinations"]

    def set_num_combinations_helper(self, config_dict):
        # gets to the total number of combinations
        # multiply across dictionary entries since you need to have every entry present in each run
        # add over list entries since you only pick one of the list values in each run
        # adds a key called num_combinations to config_dict with the total number of combinations
        # and returns it too
        num_combinations_in_list = 1
        for key, values in config_dict.items():
            # values is a list
            num_combinations = 0
            for value in values:  # value can be either a parameter value or another dict
                if type(value) is dict:
                    self.set_num_combinations_helper(value)
                    num_combinations += value["num_combinations"]
                else:
                    num_combinations += 1
                    self.keys_list.append(key)
            num_combinations_in_list *= num_combinations
        config_dict["num_combinations"] = num_combinations_in_list
        return num_combinations_in_list

    def parse(self, idx):
        # returns
        rtn_dict = dict()
        rtn_dict["run"] = int(idx / self.total_combinations)   #
        # this is the run index (for repeats)
        # For repeats, we just keep incrementing the index past total_combinations

        self.parse_helper(idx, self.config_dict, rtn_dict)

        return rtn_dict  # contains the parameters plus their values

    def parse_helper(self, idx, config_dict, rtv_dict):
        cumulative = 1
        # Populating sweep variables
        for variable, values in config_dict.items():
            if variable == "num_combinations":
                continue  # skips this key since we added it
            num_combinations = self.get_num_combinations(values)
            value, relative_idx = self.get_value_and_relative_idx(
                values, int(idx / cumulative) % num_combinations
            )
            if type(value) is dict:
                self.parse_helper(relative_idx, value, rtv_dict)  # recursive unpacking
            else:
                rtv_dict[variable] = value
            cumulative *= num_combinations


    @staticmethod
    def get_num_combinations(values):
        num_values = 0
        for value in values:
            if type(value) is dict:
                num_values += value["num_combinations"]
            else:
                num_values += 1
        return num_values

    @staticmethod
    def get_value_and_relative_idx(values, idx):
        num_values = 0
        for value in values:
            if type(value) is dict:
                temp = value["num_combinations"]
            else:
                temp = 1
            if idx < num_values + temp:
                return value, idx - num_values
            num_values += temp
        return num_values


    def convert_sbatch_idx_into_sweep_idx(self, sbatch_idx, sweep_idx_list=None):
        '''
        Takes the index incremented by sbatch and returns the matching range of sweep indices associated with that group

        This allows us to run multiple runs consecutively in the same job, which is useful if the runs are very short
        Uses the num_runs_per_group variable in the config to specify how many runs in each group

        :arg:
        sweep_idx_list: a list of indices that we want to run (a subset of all the parameter combinations)
                        if None, we run all the combinations
        :return: a list of indices to run
        '''

        start_idx = self.num_runs_per_group * sbatch_idx
        if sweep_idx_list is not None:
            return sweep_idx_list[start_idx, start_idx + self.num_runs_per_group]

        return range(start_idx, min(start_idx + self.num_runs_per_group, self.total_combinations * self.num_repeats))

    def get_num_groups_of_runs(self, sweep_idx_list=None):
        # see convert_sbatch_idx_into_sweep_idx for details

        if sweep_idx_list is not None:
            return int(len(sweep_idx_list) / self.num_runs_per_group) / self.num_runs_per_group + 1
        return int(self.total_combinations * self.num_repeats / self.num_runs_per_group) + 1

    def search(self, search_dict, only_first=True):
        """
        For any key in self.config_dict, if search_dict also has the key, use the corresponding value.
        Otherwise enumerate all values that key could take according to self.config_dict file.
        If search_dict contain any key that self.config_dict doesn't have all, that key is ignored.
        In addition, for each variable combination, list id corresponding to each run.
        For example, suppose self.total_combinations = 10 and
        we want to list ids corresponding to 4 runs, then the 5th variable combination
        corresponds to a 4-element list of ids [5, 15, 25, 35].

        :param
        search_dict: a dictionary containing key words
        only_first: returns only the first idx that matches (not repeats)
        :return: the search result,
        a list of combinations of variables related to the key words
        """
        if only_first:
            num_repeats = 1
        else:
            num_repeats = self.config_dict["num_repeats"][0]  # include it in the config file

        # find in search dict keys that don't appear in sweeper
        delete = [key for key in search_dict if key not in self.keys_set]
        temp_search_dict = search_dict.copy()
        # delete keys
        for key in delete: del temp_search_dict[key]
            
        search_result_list = []

        for idx in range(self.total_combinations):

            temp_dict = self.parse(idx)

            valid_temp_dict = True
            for key, value in temp_search_dict.items():
                if key not in temp_dict:
                    valid_temp_dict = False
                    break

            if valid_temp_dict is False:
                continue

            search_result_list.append(
                {
                    "ids": [
                        idx + run * self.total_combinations for run in range(num_repeats)
                    ]
                }
            )

            for key, value in temp_dict.items():
                if key in temp_search_dict and temp_search_dict[key] != value:
                    search_result_list = search_result_list[:-1]
                    break
                search_result_list[-1][key] = value

        return search_result_list
