# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import torch
from agents import Lineage
from parallel_solve import ParallelExecutor, mpi_fork
import argparse
import numpy as np
import logging
import sys
from es import OpenES
from simulation import battle_simulation, solve, parallel_solve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--width", default=100, type=int, help="width of the world")
    parser.add_argument("--height", default=100, type=int, help="height of the world")
    for lineage in Lineage:
        parser.add_argument("--" + lineage.name.lower(), default=0, type=int, help="number of {}".format(lineage.name))
    parser.add_argument("--mode", default="random", type=str, help="run mode")
    parser.add_argument("--iterations", default=500, type=int, help="solver iterations")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--np", default=1, type=int, help="number of parallel workers")

    args = parser.parse_args()
    n_params = 209  # 82 + (4 * 3 * 5 * 5) * 2 + 8
    set_seed(args.seed)
    if args.mode == "random":
        battle_simulation(args, np.random.random(n_params), render=True)
    elif args.mode.startswith("opt"):
        solver = OpenES(n_params,  # number of model parameters
                        sigma_init=0.5,  # initial standard deviation
                        sigma_decay=0.999,  # don't anneal standard deviation
                        learning_rate=0.1,  # learning rate for standard deviation
                        learning_rate_decay=1.0,  # annealing the learning rate
                        popsize=40,  # population size
                        antithetic=False,  # whether to use antithetic sampling
                        weight_decay=0.00,  # weight decay coefficient
                        rank_fitness=False,  # use rank rather than fitness numbers
                        forget_best=False)
        if "parallel" in args.mode:
            best = parallel_solve(solver, args.iterations, args)
        else:
            best = solve(solver, args.iterations, args)
        logging.warning("fitness score at this local optimum: {}".format(best[1]))
    elif args.mode == "opt-parallel":
        if "parent" == mpi_fork(args.n + 1):
            sys.exit()
        ParallelExecutor(args, n_params, 1, 0, args.n, 1, False, False, False, args.seed, 0.10, 0.999)
    elif args.mode == "best":
        best = np.load("best.npy")
        battle_simulation(args, best, render=True)
