# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from environment import Environment
from agents import Lineage
import argparse


def battle_simulation(args, render=False):
    done = False
    env = Environment(vars(args))
    while not done:
        print(env)
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--width", default=100, type=int, help="width of the world")
    parser.add_argument("--height", default=100, type=int, help="height of the world")
    for lineage in Lineage:
        parser.add_argument("--" + lineage.name.lower(), default=0, type=int, help="number of {}".format(lineage.name))

    args = parser.parse_args()
    battle_simulation(args, render=True)
