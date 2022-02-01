
from environment import Environment
import logging
import numpy as np
import time
from multiprocessing import Pool


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
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        np.save("best.npy", result[0])
    return result


def parallel_solve(solver, iterations, args):
    num_workers = args.np
    if solver.popsize % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    history = []
    result = None
    for j in range(iterations):
        solutions = solver.ask()
        with Pool(num_workers) as pool:
            results = pool.map(parallel_wrapper, [(args, solutions, i) for i in range(solver.popsize)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        np.save("best.npy", result[0])
    return result


def battle_simulation(args, solution, render=False, threshold=float("inf")):
    done = False
    env = Environment(vars(args), solution)
    start = time.time()
    while not done and time.time() - start < threshold:
        # print(env, time.time() - start)
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
    # logging.warning("fitness: {}".format(env.get_reward()))
    return env.get_reward()


def parallel_wrapper(args):
    args, solutions, i = args
    return i, battle_simulation(args, solutions[i], render=False, threshold=10)
