import abc
import random
from enum import Enum
import keras


ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class Lineage(Enum):
    ELF = (5, 3, 4, 5, False, "green")
    MAN = (3, 3, 4, 3, False, "red")
    URUK = (3, 4, 5, 3, True, "blue")
    ORC = (2, 3, 4, 2, True, "black")

    def __init__(self, skill, strength, defense, courage, is_evil, color):
        self.skill = skill
        self.strength = strength
        self.defense = defense
        self.courage = courage
        self.is_evil = is_evil
        self.color = color


class MLP(object):

    def __init__(self):


class BaseAgent(ABC):

    def __init__(self, i, x, y, lineage):
        self.id = i
        self.x = x
        self.y = y
        self.lineage = lineage
        self.alive = True
        self.opponent = None

    def __str__(self):
        return "Base{}[x={},y={}]".format(self.lineage.name, self.x, self.y)

    @abc.abstractmethod
    def act(self, obs):
        pass

    def move(self, x, y):
        self.x = x
        self.y = y

    def die(self):
        self.alive = False

    def is_idle(self):
        return not self.alive or self.opponent is not None

    @classmethod
    def agent_factory(cls, solution, i, x, y, lineage):
        if solution is None:
            return RandomAgent(i, x, y, lineage)
        return


class RandomAgent(BaseAgent):

    def __init__(self, i, x, y, lineage):
        super(RandomAgent, self).__init__(i, x, y, lineage)

    def __str__(self):
        return super(RandomAgent, self).__str__().replace("Base", "Random")

    def act(self, obs):
        return random.random() * 2 - 1, random.random() * 2 - 1


class GreedyAgent(BaseAgent):

    def __init__(self, i, x, y, lineage):
        super(GreedyAgent, self).__init__(i, x, y, lineage)

    def __str__(self):
        return super(GreedyAgent, self).__str__().replace("Base", "Greedy")

    def act(self, obs):
        x, y = (obs[0], obs[1]) if self.lineage.is_evil else (obs[2], obs[3])
        x -= self.x
        y -= self.y
        return min(x, 1.0) if x >= 0 else max(x, -1.0), min(y, 1.0) if y >= 0 else max(y, -1.0)


class MLPAgent(BaseAgent):

    def __init__(self, i, x, y, lineage):
        super(MLPAgent, self).__init__(i, x, y, lineage)

    def __str__(self):
        return super(MLPAgent, self).__str__().replace("Base", "MLP")

    def act(self, obs):
        pass