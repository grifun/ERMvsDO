from game import *
import scipy.optimize as sp


def compute_matrix(a_s:np.array, b_s:np.array, u:callable) -> np.array:
    # Computes the utility matrix of the reduced game
    matrix = np.zeros( (len(a_s), len(b_s)) )
    for i in range( len(a_s) ):
        for j in range( len(b_s) ):
            matrix[i,j] = u(a_s[i], b_s[j])[0]
    return matrix

def optimal_mixed_strategy(matrix:np.array, player='a', lp_solver="highs") -> np.array:
    # Computes the best strategy on the reduced game
    if player == 'a':
        matrix = matrix.transpose()
    height, width = matrix.shape
    # [1 0 0 0 ... 0]
    function_vector = np.insert( np.zeros(width), 0, 1)
    # [-1 | A]
    boundary_matrix = np.insert(matrix, 0, values=-1, axis=1)
    # [0 1 1 ... 1]
    eq_matrix = np.array([np.insert(np.ones(width), 0, values=0, axis=0)])
    # [ [-inf,inf], [0,inf]...[0,inf]]
    bnds = np.ones([width+1,2]) * np.array([0, np.inf])
    bnds[0] = np.array([-np.inf, np.inf])
    # {options} added on behalf what the functions itself demanded in stdout
    if player == 'a': #maximizing player
        ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'maxiter':int(1e5)})
    else:             #minimizing player
        ret = sp.linprog( function_vector, boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'maxiter':int(1e5)})
    if ret['success'] is not True:
        raise "DID NOT FIND EQUILIBRIUM!"

    x = ret['x'][1:]
    return x

def already_exists(xs:np.array, x:np.array, eps:float) -> bool:
    # Check if a vector x (e.g. strategy) already exists in list of xs (with epsilon tolerance)
    exists = False
    for i in range(len(xs)):
        if all( np.abs(xs[i]-x)<=eps):
            exists = True
            break
    return exists

def reduce_strategies(xs:np.array, p:np.array, ys:np.array, q:np.array, epsilon=1e-8) -> (np.array, np.array, np.array, np.array):
    # Remove strategies with small probabilities
    idx_used_a = p >= epsilon
    xs = xs[idx_used_a]
    p  = p[idx_used_a]
    p  = p / sum(p)
    idx_used_b = q >= epsilon
    ys = ys[idx_used_b]
    q  = q[idx_used_b]
    q  = q / sum(q)
    return xs, p, ys, q

def bestResponseOracleA(b_s:ActionSet, weigths:np.array, game:Game):
    # Find best response of player A against mixed strategy of B
    a0 = game.A.getRandomPoint()
    if len(a0) == 1:
        a0 = a0[0]
    a_response = sp.minimize(game.mixed_utility_function_a, a0, method='Powell', bounds=game.A.bounds, args=(b_s, weigths)).x
    #a_response = sp.minimize(game.mixed_utility_function_a, a0, bounds=A.bounds, args=(b_s, weigths)).x
    a_val = -game.mixed_utility_function_a(a_response, b_s, weigths)
    return a_response, a_val

def bestResponseOracleB(a_s:ActionSet, weigths:np.array, game:Game):
    # Find best response of player B against mixed strategy of A
    b0 = game.B.getRandomPoint()
    if len(b0) == 1:
        b0 = b0[0]
    b_response = sp.minimize(game.mixed_utility_function_b, b0, method='Powell', bounds=game.B.bounds, args=(a_s, weigths)).x
    #b_response = sp.minimize(game.mixed_utility_function_b, b0, bounds=B.bounds, args=(a_s, weigths)).x
    b_val = game.mixed_utility_function_b(b_response, a_s, weigths)
    return b_response, b_val

def valueOracle(a_s:np.array, b_s:np.array, game:Game):
    # Find the strategies in an equilibrium of player's lists of actions
    matrix = compute_matrix(a_s, b_s, game.u)
    p = optimal_mixed_strategy(matrix, player='a', lp_solver="highs")
    q = optimal_mixed_strategy(matrix, player='b', lp_solver="highs")
    return game.value_in_strategies(a_s, p, b_s, q), p, q
