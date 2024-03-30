from abc import ABCMeta
from abc import abstractmethod

from typing import List

import numpy as np
import pandas as pd

from env.bandit import GaussianBanditEnv

from tqdm import tqdm
import pdb


class ActionValueMethod(metaclass=ABCMeta):
    def __init__(self, env):
        self.env = env
        self.arm_num = self.env.arm_num
        np.random.seed(None)

    @abstractmethod
    def get_optimal_arm():
        raise NotImplementedError

    @abstractmethod
    def optimize():
        raise NotImplementedError

    def get_reward_histroy(self) -> List:
        return self._reward_history

    def get_arm_history(self) -> List:
        return self._arm_history



class EpsilonGreedy(ActionValueMethod):
    def __init__(
        self,
        env,
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


class ExpRecencyWeightAvarage(ActionValueMethod):
    """
    直近のデータを重視する。
    """
    def __init__(
        self,
        env,
        epsilon: float = 0.1,
        alpha: float = 0.1
        ):
        super().__init__(env=env)
        self.epsilon = epsilon
        self.alpha = alpha

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
            Q[optimal_arm] += self.alpha * (reward - Q[optimal_arm])

            self._arm_history.append(optimal_arm)
            self._reward_history.append(reward)

            optimal_arm = self.get_optimal_arm(Q=Q)



class UCB(ActionValueMethod):
    """
    探索するときに、ランダムに探索せず、報酬が上がりそうなものを選ぶ
    """
    def __init__(
        self,
        env,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        C: float = 2
        ):

        super().__init__(env=env)
        self.epsilon = epsilon
        self.alpha = alpha
        self.C = C

    def get_optimal_arm(self, Q: np.array, N: np.array, time) -> int:
        if np.random.rand() > self.epsilon: # Exploit
            optimal_arm_set = np.where(Q == np.max(Q))[0]
            optimal_arm = np.random.choice(optimal_arm_set)
        else: # Explore
            ucb_Q = Q + self.C * np.sqrt(np.log(time) / (N+1))
            optimal_arm_set = np.where(ucb_Q == np.max(ucb_Q))[0]
            optimal_arm = np.random.choice(optimal_arm_set)

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
        for i in range(learning_time):
            time = i + 1
            reward = self.env.step(arm=optimal_arm)

            N[optimal_arm] += 1
            Q[optimal_arm] += self.alpha * (reward - Q[optimal_arm])

            self._arm_history.append(optimal_arm)
            self._reward_history.append(reward)

            optimal_arm = self.get_optimal_arm(Q=Q, N=N, time=time)

class GradientBandit(ActionValueMethod):
    """
    UCBなどはQという行動価値観数を推定するが、今回は行動の優先度を方策で表現し、優先度と報酬で表現された期待報酬を最適化する。
    """
    def __init__(
        self,
        env,
        alpha: float = 0.1,
        ):

        super().__init__(env=env)
        self.alpha = alpha

    def get_optimal_arm(self, H: np.array) -> int:
        optimal_arm_set = np.where(H == np.max(H))[0]
        optimal_arm = np.random.choice(optimal_arm_set)

        return optimal_arm

    def optimize(
        self,
        learning_time: int == 1000
        ):
        policy = np.array([1/self.arm_num for _ in range(self.arm_num)])
        H = np.array([0 for _ in range(self.arm_num)])

        optimal_arm = np.random.randint(0, self.arm_num)
        self._reward_history = []
        self._arm_history = []
        for i in range(learning_time):
            time = i + 1
            reward = self.env.step(arm=optimal_arm)

            self._arm_history.append(optimal_arm)
            self._reward_history.append(reward)

            mean_reward = np.mean(self._reward_history)
            optimal_arm_one_hot_vector = np.array([1 if i == optimal_arm else 0 for i in range(self.arm_num)])

            H = H + self.alpha * (reward - mean_reward) * (optimal_arm_one_hot_vector - policy)

            optimal_arm = self.get_optimal_arm(H)
