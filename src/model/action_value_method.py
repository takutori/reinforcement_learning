from abc import ABCMeta
from abc import abstractmethod

from typing import List

import numpy as np
import pandas as pd

from env.bandit import GaussianBanditEnv

from tqdm import tqdm
import pdb


class ActionValueMethod(metaclass=ABCMeta):
    def __init__(self, env: GaussianBanditEnv):
        self.env = env
        self.arm_num = self.env.arm_num
        np.random.seed(None)

    @abstractmethod
    def get_optimal_arm():
        raise NotImplementedError
    
    @abstractmethod
    def optimize():
        raise NotImplementedError
    


class EpsilonGreedy(ActionValueMethod):
    def __init__(
        self,
        env: GaussianBanditEnv,
        epsilon: float = 0.1
        ):
        super().__init__(env=env)
        self.epsilon = epsilon

    def get_optimal_arm(self, Q: np.array) -> int:
        optimal_arm_set = np.where(Q == np.max(Q))[0]
        if np.random.rand() > self.epsilon: # Exploit
            optimal_arm = np.random.choice(optimal_arm_set)
        else: # Explore
            optimal_arm = np.random.randint(0, self.arm_num)

        return optimal_arm

    def optimize(
        self,
        learning_time: int == 1000
        ):
        Q = np.array([0 for _ in range(self.arm_num)])
        N = np.array([0 for _ in range(self.arm_num)])

        optimal_arm = np.random.randint(0, self.arm_num)
        self._reward_history = []
        self._arm_history = []
        for _ in range(learning_time):
            reward = self.env.step(arm=optimal_arm)

            N[optimal_arm] += 1
            Q[optimal_arm] += (reward - Q[optimal_arm]) / N[optimal_arm]

            self._arm_history.append(optimal_arm)
            self._reward_history.append(reward)

            optimal_arm = self.get_optimal_arm(Q=Q)

    def get_reward_histroy(self) -> List:
        return self._reward_history

    def get_arm_history(self) -> List:
        return self._arm_history