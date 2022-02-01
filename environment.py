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

    def __init__(self, args, solution):
        self.width = args["width"]
        self.height = args["height"]
        self.n_agents = 0
        self.evil_agents = []
        self.good_agents = []
        for lineage in Lineage:
            if lineage.is_evil:
                self.evil_agents.extend([RandomAgent(i, random.random() * self.width, random.random() * self.height, lineage) for i in range(args[lineage.name.lower()])])
            else:
                self.good_agents.extend([MLPAgent(i, random.random() * self.width, random.random() * self.height, lineage, solution) for i in range(args[lineage.name.lower()])])
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

    def get_reward(self):
        return (self._get_n_agents_alive(False) / self._get_n_agents(False)) - (self._get_n_agents_alive(True) / self._get_n_agents(True))

    def step(self):
        for agent in self.get_alive_agents():
            if agent.opponents:
                self._fight(agent)
        self._disengage()
        done = not self._get_n_agents_alive(True) or not self._get_n_agents_alive(False)
        if not done:
            self._good_tree = kdtree.create(self._get_alive_agents(False))
            self._evil_tree = kdtree.create(self._get_alive_agents(True))
            # self._update_image()
        return done

    def _update_image(self):
        self._image.fill(1)
        agents = self._get_alive_agents(False) + self._get_alive_agents(True)
        for lineage in Lineage:
            curr_agents = list(filter(lambda x: x.lineage is lineage, agents))
            for agent in curr_agents:
                cv2.circle(self._image, (int(agent.x), int(agent.y)), radius=1, color=lineage.color, thickness=-1)

    def get_observation(self, agent):
        # half_patch_size_x, half_patch_size_y = self.width / 4, self.height / 4
        # lower_x, lower_y = self._clip(agent.x - half_patch_size_x, agent.y - half_patch_size_y)
        # upper_x, upper_y = self._clip(agent.x + half_patch_size_x, agent.y + half_patch_size_y)
        obs = []
        obs.extend([res[1] for res in self._good_tree.search_knn(agent, 4)])
        obs.extend([res[1] for res in self._evil_tree.search_knn(agent, 4)])
        obs.extend([agent.x, agent.y])
        obs = np.array(obs)
        # return self._image[int(lower_x):int(upper_x), int(lower_y):int(upper_y), :].ravel()
        self._normalize_obs(obs)
        return obs

    def _clip(self, x, y):
        return max(min(x, self.width - 1), 0), max(min(y, self.height - 1), 0)

    def _normalize_obs(self, obs):
        for i in range(len(obs) - 2):
            obs[i] /= self.max_distance
        obs[-2] /= self.width
        obs[-1] /= self.height

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
        first_opponent.opponents.append(second_opponent)
        second_opponent.opponents.append(first_opponent)

    def _disengage(self):
        for agent in self.get_alive_agents():
            agent.opponents.clear()

    def _fight(self, first_agent):
        while True:
            winners, losers, is_draw = self._roll_dice(first_agent, len(first_agent.opponents))
            if not is_draw:
                break
            best_opponent_skill = max([second_agent.lineage.skill for second_agent in first_agent.opponents])
            if first_agent.lineage.skill > best_opponent_skill:
                winners, losers = [first_agent], [random.choice(first_agent.opponents)]
                break
            elif first_agent.lineage.skill < best_opponent_skill:
                winners, losers = first_agent.opponents, [first_agent]
                break
        for winner in winners:
            for loser in losers:
                injury_threshold = self._injury_table[winner.lineage.strength - 1][loser.lineage.defense - 1]
                injury_score = random.randint(1, 6)
                if injury_score >= injury_threshold:
                    loser.die()
                    winner.move(loser.x, loser.y)

    def _flee(self, agent):
        opponent = self._find_closest_enemy(agent)[0].data
        x, y = agent.x - opponent.x, agent.y - opponent.y
        return min(x, 1.0) if x >= 0 else max(x, -1.0), min(y, 1.0) if y >= 0 else max(y, -1.0)

    @staticmethod
    def _roll_dice(first, num_seconds):
        first_score = random.randint(1, 6)
        second_scores = [random.randint(1, 6) for _ in range(num_seconds)]
        if first_score > max(second_scores):
            return [first], [random.choice(first.opponents)], False
        elif first_score < max(second_scores):
            return first.opponents, [first], False
        return [first], first.opponents, True

    def render(self):
        plt.figure(1)
        plt.clf()
        self._update_image()
        for lineage in Lineage:
            plt.scatter([], [], color=lineage.color, label=lineage.name, marker="+")
        plt.legend()
        plt.imshow(self._image)
        plt.pause(0.01)
