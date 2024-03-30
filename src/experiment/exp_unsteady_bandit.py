import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.bandit import UnsteadyBanditEnv
from model.action_value_method import EpsilonGreedy, ExpRecencyWeightAvarage, UCB, GradientBandit

from pickle_func import dumpPickle, loadPickle

import pdb
from tqdm import tqdm

def exp_unsteady_bandit():
    exp_num = 500
    arm_num = 10
    learning_time = 1000

    reward_array_dict = {
        "epsilon_greedy" : np.zeros((learning_time, exp_num)),
        "exp_recency_weight_avarage" : np.zeros((learning_time, exp_num)),
        "UCB" : np.zeros((learning_time, exp_num)),
        "gradient_bandit" : np.zeros((learning_time, exp_num)),
        }

    for i in tqdm(range(exp_num)):
        env = UnsteadyBanditEnv(arm_num=arm_num)

        # epsilon = 0.1
        optimizer = EpsilonGreedy(
            env=env,
            epsilon=0.1
            )
        optimizer.optimize(
            learning_time=learning_time
        )
        reward_array_dict["epsilon_greedy"][:, i] = optimizer.get_reward_histroy()

        # epsilon = 0.1, alpha = 0.1
        optimizer = ExpRecencyWeightAvarage(
            env=env,
            epsilon=0.1,
            alpha=0.1
            )
        optimizer.optimize(
            learning_time=learning_time
        )
        reward_array_dict["exp_recency_weight_avarage"][:, i] = optimizer.get_reward_histroy()

        # epsilon = 0.1, C=2
        optimizer = UCB(
            env=env,
            epsilon=0.1,
            C=2
            )
        optimizer.optimize(
            learning_time=learning_time
        )
        reward_array_dict["UCB"][:, i] = optimizer.get_reward_histroy()

        # alpha = 0.1
        optimizer = GradientBandit(
            env=env,
            alpha=0.1
            )
        optimizer.optimize(
            learning_time=learning_time
        )
        reward_array_dict["gradient_bandit"][:, i] = optimizer.get_reward_histroy()

    dumpPickle("data/reward_data/SteadyBandit.pickle", reward_array_dict)
