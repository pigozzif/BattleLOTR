# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from environment import Environment
from agents import Lineage
import argparse
import numpy as np
import logging

from es import OpenES


def solve(solver, iterations, args):
    history = []
    result = None
    for j in range(iterations):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = battle_simulation(args, solutions[i], render=False)
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j + 1) % 10 == 0:
            logging.info("fitness at iteration {}: {}".format(j + 1, result[1]))
    return result


def battle_simulation(args, solution, render=False):
    done = False
    env = Environment(vars(args), solution)
    while not done:
        # print(env)
        obs = []
        for agent in env.get_alive_agents():
            obs.append(env.get_observation(agent))
        i = 0
        for agent in env.get_alive_agents():
            env.set_action(agent, obs[i])
            i += 1
        done = env.step()
        if render:
            env.render()
    return env.get_reward()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--width", default=100, type=int, help="width of the world")
    parser.add_argument("--height", default=100, type=int, help="height of the world")
    for lineage in Lineage:
        parser.add_argument("--" + lineage.name.lower(), default=0, type=int, help="number of {}".format(lineage.name))
    parser.add_argument("--mode", default="random", type=str, help="run mode")
    parser.add_argument("--iterations", default=500, type=int, help="solver iterations")

    args = parser.parse_args()
    n_params = 132
    if args.mode == "random":
        battle_simulation(args, np.random.random(n_params), render=True)
    elif args.mode == "opt":
        solver = oes = OpenES(n_params,  # number of model parameters
                              sigma_init=0.5,  # initial standard deviation
                              sigma_decay=0.999,  # don't anneal standard deviation
                              learning_rate=0.1,  # learning rate for standard deviation
                              learning_rate_decay=1.0,  # annealing the learning rate
                              popsize=40,  # population size
                              antithetic=False,  # whether to use antithetic sampling
                              weight_decay=0.00,  # weight decay coefficient
                              rank_fitness=False,  # use rank rather than fitness numbers
                              forget_best=False)
        best = solve(solver, args.iterations, args)
        logging.info("fitness score at this local optimum: {}".format(best[1]))
        np.save("best.npy", best[1])
    elif args.mode == "best":
        best = np.load("best.npy")
        battle_simulation(args, best, render=True)
