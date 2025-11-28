import random
from world import World

class RandomController:

    def __init__(self):
        pass

    def getActions(self, world):
        actions = {}
        for b in world.stations:
            for c in b.cells:
                actions[c.id] = bool(random.randint(0,1))

        return actions

class BooleanController:

    def __init__(self, val):
        self.val = val

    def getActions(self, world):
        actions = {}
        for b in world.stations:
            for c in b.cells:
                actions[c.id] = self.val

        return actions
