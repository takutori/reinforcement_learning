from typing import List, Union
import numpy as np
import pandas as pd



class GaussianBanditEnv:
    def __init__(
        self,
        arm_num: str,
        seed: Union[int, None] = None,
        ) -> None:

        self._fixed_seed(seed) # seed値の固定
        self.arm_num = arm_num
        self.true_rewards = np.random.rand(self.arm_num) * 10
        self.true_sigmas = np.array([1 for _ in range(self.arm_num)])

    def _fixed_seed(self, seed):
        np.random.seed(seed)

    def set_true_reward(self, rewards: np.array):
        if len(rewards) == self.arm_num:
            self.true_rewards = rewards
        else:
            print("Error: lenght of rewarads is not equal to number of arm")

    def set_true_sigmas(self, sigmas: np.array):
        if len(sigmas) == self.arm_num:
            self.sigmas = sigmas
        else:
            print("Error: lenght of sigmas is not equal to number of arm")


    def step(self, arm) -> Union[float, None]:
        try:
            reward = np.random.normal(self.true_rewards[arm], self.true_sigmas[arm])
            return reward
        except IndexError as e:
            print("IndexError: arm is out ob bounds length of arm_num")


class UnsteadyBanditEnv:
    def __init__(
        self,
        arm_num: str,
        seed: Union[int, None] = None,
        ) -> None:

        self._fixed_seed(seed) # seed値の固定
        self.arm_num = arm_num
        self.true_rewards = np.random.rand(self.arm_num) * 10
        self.true_sigmas = np.array([1 for _ in range(self.arm_num)])
        self.amount_of_changes = np.array([
            np.random.choice([1, -1]) * np.random.rand() * 2 for _ in range(self.arm_num)
            ])
        
        self.time = 0

    def _fixed_seed(self, seed):
        np.random.seed(seed)

    def set_true_reward(self, rewards: np.array):
        if len(rewards) == self.arm_num:
            self.true_rewards = rewards
        else:
            print("Error: lenght of rewarads is not equal to number of arm")

    def set_true_sigmas(self, sigmas: np.array):
        if len(sigmas) == self.arm_num:
            self.sigmas = sigmas
        else:
            print("Error: lenght of sigmas is not equal to number of arm")


    def step(self, arm: int) -> Union[float, None]:
        self.true_rewards += self.time * self.amount_of_changes
        try:
            reward = np.random.normal(self.true_rewards[arm], self.true_sigmas[arm])
            
            self.time += 1
            return reward
        except IndexError as e:
            print("IndexError: arm is out ob bounds length of arm_num")

