from game import *
import scipy.optimize as sp


def compute_matrix(a_s, b_s, u:callable):
    # Computes the utility matrix of the reduced game
    matrix = np.zeros( (len(a_s), len(b_s)) )
    for i in range( len(a_s) ):
        for j in range( len(b_s) ):
            matrix[i,j] = u(a_s[i], b_s[j])[0]
    return matrix

def optimal_mixed_strategy(matrix, player='a', lp_solver="simplex"):
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
        ret = sp.linprog( -function_vector, -boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'autoscale': True, 'sym_pos':False, 'maxiter':int(1e5)})
    else:             #minimizing player
        ret = sp.linprog( function_vector, boundary_matrix, np.zeros(height), eq_matrix, np.array([1]), bnds, method=lp_solver, options={'autoscale': True, 'sym_pos':False, 'maxiter':int(1e5)})
    if ret['success'] is not True:
        raise "DID NOT FIND EQUILIBRIUM!"

    x = ret['x'][1:]
    return x

def already_exists(xs, x, eps):
    # Check if a vector x (e.g. strategy) already exists in list of xs (with epsilon tolerance)
    exists = False
    for i in range(len(xs)):
        if all( np.abs(xs[i]-x)<=eps):
            exists = True
            break
    return exists

def reduce_strategies(xs, p, ys, q, epsilon=1e-8):
    # Remove strategies with small probabilities
    ii = p >= epsilon
    xs = xs[ii]
    p  = p[ii]
    p  = p / sum(p)
    jj = q >= epsilon
    ys = ys[jj]
    q  = q[jj]
    q  = q / sum(q)
    return xs, p, ys, q

def bestResponseOracleA(A:HyperBlock, b_s:ActionSet, weigths:np.array, game:Game, eps:float):
    a0 = A.getRandomPoint()[0]
    a_response = sp.minimize(game.mixed_utility_function_a, a0, method='Powell', bounds=A.bounds, args=(b_s, weigths)).x
    a_val = -game.mixed_utility_function_a(a_response, b_s, weigths)
    return a_response, a_val

def bestResponseOracleB(B:HyperBlock, a_s:ActionSet, weigths:np.array, game:Game, eps:float):
    b0 = B.getRandomPoint()[0]
    b_response = sp.minimize(game.mixed_utility_function_b, b0, method='Powell', bounds=B.bounds, args=(a_s, weigths)).x
    b_val = game.mixed_utility_function_b(b_response, a_s, weigths)
    return b_response, b_val

def value_oracle(a_s, b_s, game:Game):
    matrix = compute_matrix(a_s, b_s, game.u)
    p = optimal_mixed_strategy(matrix, player='a', lp_solver="highs")
    q = optimal_mixed_strategy(matrix, player='b', lp_solver="highs")
    return game.value_in_strategies(a_s, p, b_s, q), p, q
