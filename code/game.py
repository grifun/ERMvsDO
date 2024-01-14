import numpy as np
import random
import copy

class Space:
    # abstract class
    def getRandomPoint(self):
        raise "not implemented yet"

class HyperBlock(Space):
    # hypercube of dimension n, inited with array of bounds (lb,up) for all dimensions
    def __init__(self, bounds: np.array):
        self.n = len(bounds)
        self.bounds = bounds

    def getRandomPoint(self) -> np.array:
        return np.array([ np.array( self.bounds @ np.array([-1,1]) * np.random.rand(self.n) + self.bounds[:,0] ) ])

    def getBounds(self) -> np.array:
        return copy.deepcopy(self.bounds.T)

class ActionSet(Space):
    # list of actions, inited as numpy array
    def __init__(self, actions: np.array):
        self.actions = actions

    def getRandomPoint(self) -> np.array:
        return random.choice(self.actions)


class Game:
    # Definition of a game, containing ActionSpace for both players, and the function defining the game
    def __init__(self, A: HyperBlock, B: HyperBlock, u: callable, name:str=""):
        self.u = u
        self.A = A
        self.B = B
        self.name = name
    
    def mixed_utility_function_a(self, a, b_s, q):
        # Compute the utility function of player 1.
        us = np.array([self.u(a,b) for b in b_s])
        val = -q @ us
        if isinstance(val, np.ndarray):
            if len(val) > 1:
                raise "something wrong"
            else:
                val = val[0]
        return val

    def mixed_utility_function_b(self, b, a_s, p):
        # Computed the utility function of player 2.
        us = np.array([self.u(a,b) for a in a_s])
        val = p @ us
        if isinstance(val, np.ndarray):
            if len(val) > 1:
                raise "something wrong"
            else:
                val = val[0]
        return val

    def value_in_strategies(self, a_s, p, b_s, q):
        # Computed the utility function in both player's specified strategies
        temp_vals = np.array( [ -self.mixed_utility_function_a(a, b_s, q) for a in a_s ] )
        return temp_vals @ p