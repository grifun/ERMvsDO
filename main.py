from funcs import *
from game import *
from double_oracle import *
from erm import *

games = [
    Game(A=HyperBlock( np.array([[-1.,1.]]) ), B=HyperBlock( np.array([[-1.,1.]]) ), u=util1, name="5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) - y"),
    Game(A=HyperBlock( np.array([[-2.25,2.5]]) ), B=HyperBlock( np.array([[-2.5,1.75]]) ), u=util2, name="Rosenbrock"),
    Game(A=HyperBlock( bounds_f1_A ), B=HyperBlock( bounds_f1_B ), u=util_f1, name="example_f1"),
    Game(A=HyperBlock( bounds_f2_A ), B=HyperBlock( bounds_f2_A ), u=util_f2, name="example_f2"),
    Game(A=HyperBlock( bounds_f_example1_A ), B=HyperBlock( bounds_f_example1_B), u=util_f_example1, name="example1"),
    Game(A=HyperBlock( bounds_f_example4_A ), B=HyperBlock( bounds_f_example4_B), u=util_f_example4, name="example4"),
    Game(A=HyperBlock( bounds_f_example5_A ), B=HyperBlock( bounds_f_example5_B), u=util_f_example5, name="example5"),
    Game(A=HyperBlock( bounds_f_example6_A ), B=HyperBlock( bounds_f_example6_B), u=util_f_example6, name="example6"),
    Game(A=HyperBlock( bounds_f_example8_A ), B=HyperBlock( bounds_f_example8_B), u=util_f_example8, name="example8"),
    Game(A=HyperBlock( bounds_f_example9_A ), B=HyperBlock( bounds_f_example9_B), u=util_f_example9, name="example9"),
    Game(A=HyperBlock( bounds_f_example10_A ), B=HyperBlock( bounds_f_example10_B), u=util_f_example10, name="example10"),
    Game(A=HyperBlock( bounds_f_example11_A ), B=HyperBlock( bounds_f_example11_B), u=util_f_example11, name="(x-y)**2"),
    Game(A=HyperBlock( bounds_f_example12_A ), B=HyperBlock( bounds_f_example12_B), u=util_f_example12, name="x**2 - 2*x*y + y**2")
]

def townsend(x,y):
    term1 = np.power( np.cos( (x-0.1)*y ), 2)
    term2 = x*np.sin( 3*x+y )
    return -term1 -term2

A_bounds = np.array([[-2.25, 2.5]])  #interval (-2.5;2.5) in one dimension only
B_bounds = np.array([[-2.25, 1.75]]) #interval (-2.5;1.75) in one dimension only

game_townsend = Game(A=HyperBlock(A_bounds), B=HyperBlock(B_bounds), u=townsend, name="townsend")

eps = 1e-6
C   = 1e-11
init_algorithm = "bounds"

from double_oracle import *
xs, p, ys, q, value_lbs, value_ubs = eps_nash_erm(game=game_townsend, init_algorithm=init_algorithm, maxiter=20, eps=eps)
DO_outcome = (xs, p, ys, q, value_lbs, value_ubs)
