from agents import *
import random
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt


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
        self.good_tree = KDTree(self._get_alive_agents_positions(False))
        self.evil_tree = KDTree(self._get_alive_agents_positions(True))
        self.max_distance = euclidean([0, 0], [self.width, self.height])

    def __str__(self):
        return "Env[n_good_agents={},n_evil_agents={}]".format(self._get_n_agents_alive(False), self._get_n_agents_alive(True))

    def __iter__(self):
        self._i = 0
        return self

    def next(self):
        if self._i >= len(self.good_agents) + len(self.evil_agents):
            raise StopIteration
        agents = self.good_agents + self.evil_agents
        agent = agents[self._i]
        while not agent.alive:
            self._i += 1
            if self._i >= len(agents):
                raise StopIteration
            agent = agents[self._i]
        self._i += 1
        return agent

    def _get_n_agents(self, evil):
        return len(self.evil_agents) if evil else len(self.good_agents)

    def _get_n_agents_alive(self, evil):
        return len(self.evil_tree.data) if evil else len(self.good_tree.data)

    def _get_alive_agents(self, evil):
        if evil:
            return np.array([agent for agent in filter(lambda x: x.lineage.is_evil, self)])
        return np.array([agent for agent in filter(lambda x: not x.lineage.is_evil, self)])

    def _get_alive_agents_positions(self, evil):
        return np.array([[agent.x, agent.y] for agent in self._get_alive_agents(evil)])

    def step(self):
        for agent in self:
            if agent.opponent is not None:
                self._fight(agent, agent.opponent)
        done = not len(self._get_alive_agents(True)) or not len(self._get_alive_agents(False))
        if not done:
            self.good_tree = KDTree(self._get_alive_agents_positions(False))
            self.evil_tree = KDTree(self._get_alive_agents_positions(True))
        return done

    def get_observation(self, agent):
        pos = [agent.x, agent.y]
        good = self.good_tree.query(pos)
        evil = self.evil_tree.query(pos)
        obs = np.array([self.good_tree.data[good[1]][0],
                        self.good_tree.data[good[1]][1],
                        self.evil_tree.data[evil[1]][0],
                        self.evil_tree.data[evil[1]][1],
                        len(self.good_tree.query_ball_point(pos, 10.0)),
                        len(self.evil_tree.query_ball_point(pos, 10.0)),
                        agent.x,
                        agent.y,
                        0 if agent.lineage.is_evil else 1])
        self._normalize_obs(obs)
        return obs

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
            x, y = max(min(agent.x + action[0], self.width - 1), 0), max(min(agent.y + action[1], self.height - 1), 0)
        agent.move(x, y)
        closest_enemies = self.good_tree.query_ball_point([x, y], r=1.0) if agent.lineage.is_evil else self.evil_tree.query_ball_point([x, y], r=1.0)
        if not len(closest_enemies):
            return
        closest_enemies = self._get_alive_agents(not agent.lineage.is_evil)[closest_enemies]
        closest_enemies = filter(lambda a: not a.is_idle(), closest_enemies)
        closest_enemy = sorted(closest_enemies, key=lambda a: euclidean([a.x, a.y], [x, y]))[0]
        if euclidean([closest_enemy.x, closest_enemy.y], [x, y]) <= 1.0:
            self._engage(agent, closest_enemy)

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
        closest_enemy = self.good_tree.query([agent.x, agent.y]) if agent.lineage.is_evil else self.evil_tree.query([agent.x, agent.y])
        opponent = self._get_alive_agents(not agent.lineage.is_evil)[closest_enemy[1]]
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
        image = np.ones((self.width, self.height, 3))
        agents = list(self._get_alive_agents(False)) + list(self._get_alive_agents(True))
        for lineage in Lineage:
            curr_agents = list(filter(lambda x: x.lineage is lineage, agents))
            plt.scatter([agent.x for agent in curr_agents],
                        [agent.y for agent in curr_agents], color=lineage.color, label=lineage.name)
        plt.legend()
        plt.imshow(image)
        plt.pause(0.01)
