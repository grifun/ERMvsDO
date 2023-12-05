from game import *
from oracles import *

def init_ERM(game:Game, init_algorithm:str):
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
    new_strategies = np.array(strategies[:])
    for a in additions:
        if not already_exists(new_strategies, a, eps):
            new_strategies = np.insert(new_strategies, 0, values=a, axis=0)
    return new_strategies

def eps_nash_erm(game:Game, C:float = 1e-11, init_algorithm:str = "bounds", maxiter:int = 20, eps:float=1e-6):
    # separate initial actions
    a_s, b_s = init_ERM(game, init_algorithm)
    lower_bounds, upper_bounds = [],[]
    a_len_history, b_len_history = [],[]

    for t in range(maxiter):
        print("iter ", t)
        # step (a)
        b_mixed_strategy, a_responses = eps_nash_half_infinite(b_s, game.A, game, C, eps, player="a")
        # step (b)
        new_a_s = insert_strategies(a_s, a_responses, eps)
        # step (c)
        a_mixed_strategy, b_responses = eps_nash_half_infinite(new_a_s, game.B, game, C, eps, player="b")
        # step (d)
        new_b_s = insert_strategies(b_s, b_responses, eps)
        # step (e)
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

def eps_nash_half_infinite(action_list:ActionSet, action_space:HyperBlock, game:Game, C:float, eps:float, player:str="a"):
    """
    player: specifies, for which player we compute the responses AKA which player has the infinite action space
            a minimizes u
            b maximizes u AKA minimizes -u
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
        # step (a) - reweight discrete player's actions' probabilities
        if player == "a":
            utilities = np.array([ -game.u( responses[-1], action_list[i] )[0] for i in range(n) ])
        else:
            utilities = np.array([ game.u( action_list[i], responses[-1] )[0] for i in range(n) ])

        assert utilities.shape == mixed_strategies[-1].shape
        curr_mixed_strategy = mixed_strategies[-1] * np.exp( eta * utilities)
        curr_mixed_strategy /= sum(curr_mixed_strategy)

        # step (b) - get continous player's best response
        curr_response, curr_val = get_response(action_space, action_list, curr_mixed_strategy, game, eps)

        # prepare for the next iteration
        mixed_strategies.append(curr_mixed_strategy)
        responses.append(curr_response)
    return np.sum(mixed_strategies, axis=0) / len(mixed_strategies), responses