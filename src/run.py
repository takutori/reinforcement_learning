import numpy as np
import pandas as pd

import argparse
from experiment.exp_epsilon_greedy import exp_gaussian_bandit
from experiment.exp_unsteady_bandit import exp_unsteady_bandit


parser = argparse.ArgumentParser(description="")

parser.add_argument('-exp_name', '--exp_name', type=str, help="experiment name")
args = parser.parse_args()

experiment_dict = {
    "gaussian_bandit" : exp_gaussian_bandit,
    "unsteady_bandit" : exp_unsteady_bandit,
}

def main():

    experiment_dict[args.exp_name]()

if __name__ == '__main__':
    main()