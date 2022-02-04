import abc
import random
from enum import Enum
import torch
import numpy as np

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class Lineage(Enum):
    ELF = (5, 3, 4, 5, False, (0, 1, 0))
    MAN = (3, 3, 4, 3, False, (1, 0, 0))
    URUK = (3, 4, 5, 3, True, (0, 0, 1))
    ORC = (2, 3, 4, 2, True, (0, 0, 0))

    def __init__(self, skill, strength, defense, courage, is_evil, color):
        self.skill = skill
        self.strength = strength
        self.defense = defense
        self.courage = courage
        self.is_evil = is_evil
        self.color = color


class BaseAgent(ABC):

    def __init__(self, i, x, y, lineage):
        self.id = i
        self.x = x
        self.y = y
        self.lineage = lineage
        self.alive = True
        self.opponents = []
        self._move_lookup = {0: (-1, -1), 1: (0, -1), 2: (+1, +1), 3: (+1, 0), 4: (+1, -1), 5: (0, +1), 6: (-1, +1),
                             7: (-1, 0), 8: (0, 0)}

    def __str__(self):
        return "Base{}[x={},y={}]".format(self.lineage.name, self.x, self.y)

    def __len__(self):
        return 2

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("using {} index on BaseAgent".format(item))

    @abc.abstractmethod
    def act(self, obs):
        pass

    def move(self, x, y):
        self.x = x
        self.y = y

    def die(self):
        self.alive = False

    def is_idle(self):
        return not self.alive or self.opponents

    @classmethod
    def agent_factory(cls, solution, i, x, y, lineage):
        if solution is None:
            return RandomAgent(i, x, y, lineage)
        return MLPAgent(i, x, y, lineage, solution)


class RandomAgent(BaseAgent):

    def __init__(self, i, x, y, lineage):
        super(RandomAgent, self).__init__(i, x, y, lineage)

    def __str__(self):
        return super(RandomAgent, self).__str__().replace("Base", "Random")

    def act(self, obs):
        return random.choice(list(self._move_lookup.values()))


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

    def __init__(self, i, x, y, lineage, solution):
        super(MLPAgent, self).__init__(i, x, y, lineage)
        self.nn = torch.nn.Sequential(torch.nn.Linear(in_features=10, out_features=10), torch.nn.Tanh(),
                                      torch.nn.Linear(in_features=10, out_features=9), torch.nn.Softmax(dim=0)
                                      # torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(5, 5)),
                                      # torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      # torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5, 5)),
                                      # torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      # torch.nn.Linear(in_features=40, out_features=2), torch.nn.Tanh()
                                      )
        self.set_params(solution)
        for param in self.nn.parameters():
            param.requires_grad = False

    def __str__(self):
        return super(MLPAgent, self).__str__().replace("Base", "MLP")

    def set_params(self, params):
        state_dict = self.nn.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(params[start:start + num])
            start += num

    def act(self, obs):
        return self._move_lookup[int(np.argmax(self.nn(torch.from_numpy(obs).float()).detach().numpy()))]
