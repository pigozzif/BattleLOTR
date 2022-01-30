from agents import *
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import kdtree
import cv2


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class Environment(object):

    def __init__(self, args, solution=None):
        self.width = args["width"]
        self.height = args["height"]
        self.n_agents = 0
        self.evil_agents = []
        self.good_agents = []
        for lineage in Lineage:
            if lineage.is_evil:
                self.evil_agents.extend([RandomAgent(i, random.random() * self.width, random.random() * self.height, lineage) for i in range(args[lineage.name.lower()])])
            else:
                self.good_agents.extend([RandomAgent(i, random.random() * self.width, random.random() * self.height, lineage) for i in range(args[lineage.name.lower()])])
            self.n_agents += args[lineage.name.lower()]
        self._injury_table = [[3, 4, 5, 6, 7, 8],
                              [2, 3, 4, 5, 6, 7],
                              [1, 2, 3, 4, 5, 6],
                              [1, 1, 2, 3, 4, 5]]
        self._i = 0
        self._image = np.ones((self.width, self.height, 3))
        self._good_tree = kdtree.create(self._get_alive_agents(False))
        self._evil_tree = kdtree.create(self._get_alive_agents(True))
        self.max_distance = euclidean([0, 0], [self.width, self.height])

    def __str__(self):
        return "Env[n_good_agents={},n_evil_agents={}]".format(self._get_n_agents_alive(False), self._get_n_agents_alive(True))

    def _get_n_agents(self, evil):
        return len(self.evil_agents) if evil else len(self.good_agents)

    def _get_n_agents_alive(self, evil):
        return len(self._get_alive_agents(True)) if evil else len(self._get_alive_agents(False))

    def _get_alive_agents(self, evil):
        if evil:
            return [agent for agent in filter(lambda x: x.lineage.is_evil and x.alive, self.evil_agents)]
        return [agent for agent in filter(lambda x: not x.lineage.is_evil and x.alive, self.good_agents)]

    def _get_alive_agents_positions(self, evil):
        return np.array([[agent.x, agent.y] for agent in self._get_alive_agents(evil)])

    def get_alive_agents(self):
        return self._get_alive_agents(True) + self._get_alive_agents(False)

    def step(self):
        for agent in self.get_alive_agents():
            if agent.opponent is not None:
                self._fight(agent, agent.opponent)
        done = not self._get_n_agents_alive(True) or not self._get_n_agents_alive(False)
        if not done:
            self._good_tree = kdtree.create(self._get_alive_agents(False))
            self._evil_tree = kdtree.create(self._get_alive_agents(True))
            self._update_image()
        return done

    def _update_image(self):
        self._image.fill(1)
        agents = self._get_alive_agents(False) + self._get_alive_agents(True)
        for lineage in Lineage:
            curr_agents = list(filter(lambda x: x.lineage is lineage, agents))
            for agent in curr_agents:
                cv2.circle(self._image, (int(agent.x), int(agent.y)), radius=1, color=lineage.color, thickness=-1)

    def get_observation(self, agent):
        half_patch_size_x, half_patch_size_y = self.width / 4, self.height / 4
        lower_x, lower_y = self._clip(agent.x - half_patch_size_x, agent.y - half_patch_size_y)
        upper_x, upper_y = self._clip(agent.x + half_patch_size_x, agent.y + half_patch_size_y)
        return self._image[int(lower_x):int(upper_x), int(lower_y):int(upper_y), :].ravel()

    def _clip(self, x, y):
        return max(min(x, self.width - 1), 0), max(min(y, self.height - 1), 0)

    def _normalize_obs(self, obs):
        obs[0] /= self.max_distance
        obs[1] /= self.max_distance
        obs[2] /= len(self.good_agents)
        obs[3] /= len(self.evil_agents)
        obs[4] /= self.width
        obs[5] /= self.height

    def set_action(self, agent, obs):
        if agent.is_idle():
            return
        elif self._get_n_agents_alive(agent.lineage.is_evil) < 0.1 * self._get_n_agents(agent.lineage.is_evil) and random.randint(1, 6) + agent.lineage.courage < 10:
            x, y = self._flee(agent)
        else:
            action = agent.act(obs)
            x, y = self._clip(agent.x + action[0], agent.y + action[1])
        agent.move(x, y)
        closest_enemy, distance = self._find_closest_enemy(agent)
        if distance <= 1.0 and not closest_enemy.data.is_idle():
            self._engage(agent, closest_enemy.data)

    def _find_closest_enemy(self, agent):
        return self._good_tree.search_nn(agent) if agent.lineage.is_evil else self._evil_tree.search_nn(agent)

    @staticmethod
    def _engage(first_opponent, second_opponent):
        first_opponent.opponent = second_opponent
        second_opponent.opponent = first_opponent

    @staticmethod
    def _disengage(first_opponent, second_opponent):
        first_opponent.opponent = None
        second_opponent.opponent = None

    def _fight(self, first_agent, second_agent):
        winner, loser, is_draw = self._roll_dice(first_agent, second_agent)
        if is_draw:
            if first_agent.lineage.skill == second_agent.lineage.skill:
                is_again_draw = True
                while is_again_draw:
                    winner, loser, is_again_draw = self._roll_dice(first_agent, second_agent)
            else:
                loser, winner = tuple(sorted([first_agent, second_agent], key=lambda x: x.lineage.skill))
        injury_threshold = self._injury_table[winner.lineage.strength - 1][loser.lineage.defense - 1]
        injury_score = random.randint(1, 6)
        if injury_score >= injury_threshold:
            loser.die()
            winner.move(loser.x, loser.y)
        self._disengage(first_agent, second_agent)

    def _flee(self, agent):
        opponent = self._find_closest_enemy(agent)[0].data
        x, y = agent.x - opponent.x, agent.y - opponent.y
        return min(x, 1.0) if x >= 0 else max(x, -1.0), min(y, 1.0) if y >= 0 else max(y, -1.0)

    @staticmethod
    def _roll_dice(first, second):
        first_score = random.randint(1, 6)
        second_score = random.randint(1, 6)
        if first_score > second_score:
            return first, second, False
        elif first_score < second_score:
            return second, first, False
        return first, second, True

    def render(self):
        plt.figure(1)
        plt.clf()
        for lineage in Lineage:
            plt.scatter([], [], color=lineage.color, label=lineage.name)
        plt.legend()
        plt.imshow(self._image)
        plt.pause(0.01)
