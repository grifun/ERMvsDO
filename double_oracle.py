from game import *
from oracles import *
import numpy as np

def init_DO(game:Game, init_algorithm:str):
    if init_algorithm == "bounds":
        a_s = game.A.getBounds()
        b_s = game.B.getBounds()
    elif init_algorithm == "random":
        a_s = game.A.getRandomPoint()
        b_s = game.A.getRandomPoint()
    else:
        raise "Unsupported init algorithm"
    p = np.ones(len(a_s))/len(a_s)
    q = np.ones(len(b_s))/len(b_s)
    return a_s, p, b_s, q

def double_oracle(game, init_algorithm:str = "bounds", maxiter=20, eps=1e-6):
    # Initialize the algorithm
    a_s, p, b_s, q = init_DO(game, init_algorithm)
    lower_bounds, upper_bounds = [],[]

    for itr in range(maxiter):
        print("itr: ", itr)
        # Find best pure response
        a, a_opt_val = bestResponseOracleA(game.A, b_s, q, game, eps)
        b, b_opt_val = bestResponseOracleB(game.B, a_s, p, game, eps)
        upper_bounds.append( a_opt_val )
        lower_bounds.append( b_opt_val )

        # Add the best responses if they are already not found
        if not already_exists(a_s, a, eps):
            a_s = np.insert(a_s, 0, values=a, axis=0)
        if not already_exists(b_s, b, eps):
            b_s = np.insert(b_s, 0, values=b, axis=0)

        # Recompute the matrix of the reduced game
        matrix = compute_matrix(a_s, b_s, game.u)

        # Find the best strategies on the reduced game
        p = optimal_mixed_strategy(matrix, player='a', lp_solver="highs")
        q = optimal_mixed_strategy(matrix, player='b', lp_solver="highs")

        if abs(upper_bounds[-1] - lower_bounds[-1]) < eps:
            break

    return np.flip(a_s.T).T, np.flip(p), np.flip(b_s.T).T, np.flip(q), lower_bounds, upper_bounds