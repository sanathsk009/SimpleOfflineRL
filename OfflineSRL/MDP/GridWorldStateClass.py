''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from OfflineSRL.MDP.old_State import State
import numpy as np

class GridWorldState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y])
        self.x = round(x, 5)
        self.y = round(y, 5)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def _vrepr(self):
        return np.array([self.x,self.y])

    def __eq__(self, other):
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y