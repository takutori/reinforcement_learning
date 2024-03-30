import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.bandit import GaussianBanditEnv
from model.action_value_method import EpsilonGreedy

from pickle_func import dumpPickle, loadPickle

import pdb
from tqdm import tqdm

def exp_gaussian_bandit():
    exp_num = 2000
    arm_num = 10
    learning_time = 1000

    reward_array_dict = {
        "epsilon_greedy_0" : np.zeros((learning_time, exp_num)),
        "epsilon_greedy_0.01" : np.zeros((learning_time, exp_num)),
        "epsilon_greedy_0.1" : np.zeros((learning_time, exp_num)),
        }

    for i in tqdm(range(exp_num)):
        env = GaussianBanditEnv(arm_num=arm_num)
        
        # epsilon = 0
        epsilon_greedy = EpsilonGreedy(
            env=env,
            epsilon=0
            )
        epsilon_greedy.optimize(
            learning_time=learning_time
        )
        reward_array_dict["epsilon_greedy_0"][:, i] = epsilon_greedy.get_reward_histroy()


        # epsilon = 0.01
        epsilon_greedy = EpsilonGreedy(
            env=env,
            epsilon=0.01
            )
        epsilon_greedy.optimize(
            learning_time=learning_time
        )
        reward_array_dict["epsilon_greedy_0.01"][:, i] = epsilon_greedy.get_reward_histroy()

        # epsilon = 0.1
        epsilon_greedy = EpsilonGreedy(
            env=env,
            epsilon=0.1
            )
        epsilon_greedy.optimize(
            learning_time=learning_time
        )
        reward_array_dict["epsilon_greedy_0.1"][:, i] = epsilon_greedy.get_reward_histroy()


    dumpPickle("data/reward_data/GaussianBandit.pickle", reward_array_dict)
