from game import *
from oracles import *

def init_ERM(game:Game, init_algorithm:str):
    # Returns initial strategies for both players
    if init_algorithm == "bounds":
        a_s = game.A.getBounds()
        b_s = game.B.getBounds()
    elif init_algorithm == "random":
        a_s = game.A.getRandomPoint()
        b_s = game.A.getRandomPoint()
    else:
        raise "Unsupported init algorithm"
    return a_s, b_s

def insert_strategies(strategies: list, additions: list, eps: float):
    # Appends list of strategies to another list of strategies, if they are not already present (with epsilon-approximation)
    new_strategies = np.array(strategies[:])
    for a in additions:
        if not already_exists(new_strategies, a, eps):
            new_strategies = np.insert(new_strategies, 0, values=a, axis=0)
    return new_strategies

# Implementation follows https://arxiv.org/abs/2307.01689 , Online Learning and Solving Infinite Games with an ERM Oracle
def eps_nash_erm(game:Game, C:float = 1e-11, init_algorithm:str = "bounds", maxiter:int = 20, eps:float=1e-6):
    """Given an infinite game, iteratively find strategies in an epsilon-Nash-equilibrium.
    """
    # Separate initial actions
    a_s, b_s = init_ERM(game, init_algorithm)
    lower_bounds, upper_bounds = [],[]
    a_len_history, b_len_history = [],[]

    for t in range(maxiter):
        print("iter ", t)
        # Step (a)
        b_mixed_strategy, a_responses = eps_nash_half_infinite(b_s, game.A, game, C, eps, player="a")
        # Step (b)
        new_a_s = insert_strategies(a_s, a_responses, eps)
        # Step (c)
        a_mixed_strategy, b_responses = eps_nash_half_infinite(new_a_s, game.B, game, C, eps, player="b")
        # Step (d)
        new_b_s = insert_strategies(b_s, b_responses, eps)
        # Step (e)
        game_value1,_,_ = value_oracle(new_a_s, b_s, game)
        game_value2,_,_ = value_oracle(a_s, b_s, game)
        game_value3,a_mixed_strategy,b_mixed_strategy = value_oracle(new_a_s, new_b_s, game)
        
        lower_bounds.append( game_value3 )
        upper_bounds.append( game_value1 )

        if t > 1 and (abs(game_value1- game_value2) <= eps or abs(game_value3 - game_value2) <= eps):
            break

        a_len_history.append(len(a_s))
        b_len_history.append(len(b_s))
        a_s, b_s = new_a_s, new_b_s
    return np.flip(new_a_s.T).T, np.flip(a_mixed_strategy), np.flip(new_b_s.T).T, np.flip(b_mixed_strategy), lower_bounds, upper_bounds

# Implementation follows https://arxiv.org/abs/2307.01689 , Online Learning and Solving Infinite Games with an ERM Oracle
def eps_nash_half_infinite(action_list:ActionSet, action_space:HyperBlock, game:Game, C:float, eps:float, player:str="a"):
    """Given an infinite game and list of allowed actions of one player, iteratively find strategies in an epsilon-Nash-equilibrium.
    """
    if player == "a":
        get_response = bestResponseOracleA
    elif player == "b":
        get_response = bestResponseOracleB
    else:
        raise "invalid player"

    n = len(action_list)
    T = int( (C*np.log(n))/(eps**2) +0.5)
    print("T = ", T)
    eta = np.sqrt( np.log(n)/(2*T) )

    mixed_strategies = [ np.ones(n)/n ]

    first_response, val = get_response(action_space, action_list, mixed_strategies[-1], game, eps)
    responses = [ first_response ]
    for t in range(1, T):
        # Step (a) - reweight discrete player's actions' probabilities
        if player == "a":
            utilities = np.array([ -game.u( responses[-1], action_list[i] )[0] for i in range(n) ])
        else:
            utilities = np.array([ game.u( action_list[i], responses[-1] )[0] for i in range(n) ])

        assert utilities.shape == mixed_strategies[-1].shape
        curr_mixed_strategy = mixed_strategies[-1] * np.exp( eta * utilities)
        curr_mixed_strategy /= sum(curr_mixed_strategy)

        # Step (b) - get continous player's best response
        curr_response, curr_val = get_response(action_space, action_list, curr_mixed_strategy, game, eps)

        # Prepare for the next iteration
        mixed_strategies.append(curr_mixed_strategy)
        responses.append(curr_response)
    return np.sum(mixed_strategies, axis=0) / len(mixed_strategies), responses