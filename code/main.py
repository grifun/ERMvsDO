from funcs import *
from game import *
from double_oracle import *
from erm import *
from plotting import *
import time

# Rock - Paper - Scissors C experiment
def RPS_C_experiment():
    eps = 1e-6
    RPS = Game(A=ActionSet(RPS_actions), B=ActionSet(RPS_actions), u=utilRPS, name="RockPaperScissors")
    Cs, iterations, errors, times = [],[],[],[]
    for C in [1e-9, 8e-10, 5e-10, 2e-10, 1e-10, 5*1e-11, 1e-11, 5*1e-12]:
        time_start = time.perf_counter()
        a_mixed_strategy, b_responses = eps_nash_finite(RPS, C, eps)
        time_end = time.perf_counter()
        value = RPS.value_in_strategies(RPS.A.actions, a_mixed_strategy, b_responses, np.ones_like(b_responses)/len(b_responses))
        Cs.append(C)
        iterations.append(len(b_responses))
        errors.append(abs(value))
        times.append((time_end-time_start)*1000)
    plotCs(Cs, iterations, errors, times)

# Rosenbrock C experiment
def rosenbrock_experiment():
    eps = 1e-6
    game = Game(A=HyperBlock( bounds_3_A ), B=HyperBlock( bounds_3_B ), u=util3, name="Rosenbrock")
    xs, p, ys, q, value_lbs, value_ubs, BRs = double_oracle(game=game, init_algorithm="bounds", maxiter=10, eps=eps)
    NE_value = game.value_in_strategies(xs, p, ys, q)
    print("value found by DO: ", NE_value)
    
    NE_reached = np.inf
    Cs, errors, best_responses, times = [],[],[],[]
    C = 1e-11

    while np.abs(NE_reached - NE_value) >= NE_value / 1000:
        time_start = time.perf_counter()
        xs, p, ys, q, value_lbs, value_ubs, BRs, as_lens, bs_lens = eps_nash_erm(game=game, init_algorithm="bounds", maxiter=10, eps=eps, C=C)
        time_end = time.perf_counter()
        NE_reached = game.value_in_strategies(xs, p, ys, q)
        Cs.append(C)
        best_responses.append(BRs[-1])
        errors.append(NE_reached - NE_value)
        times.append((time_end-time_start)*1000)
        C *= 1.2
        print("Difference reached: ", np.abs(NE_reached - NE_value))
    plotCsInfinite(Cs, best_responses, errors, times, display=True)

# set the collection of games
def convergence_comparison():
    # set constants
    eps = 1e-6
    C   = 1e-11
    init_algorithm = "bounds"
    games = [
        Game(A=HyperBlock( np.array([[-1.,1.]]) ), B=HyperBlock( np.array([[-1.,1.]]) ), u=lambda x,y: 5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) - y, name="AdHoc"),
        Game(A=HyperBlock( bounds_2_A ), B=HyperBlock( bounds_2_B ), u=util2, name="Townsend"),
        Game(A=HyperBlock( bounds_3_A ), B=HyperBlock( bounds_3_B ), u=util3, name="Rosenbrock"),
        Game(A=HyperBlock( bounds_4_A ), B=HyperBlock( bounds_4_B ), u=util4, name="util4"),
        Game(A=HyperBlock( bounds_f_example1_A ), B=HyperBlock( bounds_f_example1_B), u=util_f_example1, name="example1"),
        Game(A=HyperBlock( bounds_f_example2_A ), B=HyperBlock( bounds_f_example2_B), u=util_f_example2, name="example2"),
        Game(A=HyperBlock( bounds_f_example3_A ), B=HyperBlock( bounds_f_example3_B), u=util_f_example3, name="example3"),
        Game(A=HyperBlock( bounds_f_example4_A ), B=HyperBlock( bounds_f_example4_B), u=util_f_example4, name="example4"),
        Game(A=HyperBlock( bounds_f_example5_A ), B=HyperBlock( bounds_f_example5_B), u=util_f_example5, name="example5"),
        Game(A=HyperBlock( bounds_f_example6_A ), B=HyperBlock( bounds_f_example6_B), u=util_f_example6, name="example6"),
        Game(A=HyperBlock( bounds_f_example7_A ), B=HyperBlock( bounds_f_example7_B), u=util_f_example7, name="example7"),
        Game(A=HyperBlock( bounds_f_example8_A ), B=HyperBlock( bounds_f_example8_B), u=util_f_example8, name="example8"),
        Game(A=HyperBlock( bounds_f_example9_A ), B=HyperBlock( bounds_f_example9_B), u=util_f_example9, name="example9"),
        Game(A=HyperBlock( bounds_f_example10_A ), B=HyperBlock( bounds_f_example10_B), u=util_f_example10, name="example10"),
        Game(A=HyperBlock( bounds_f_example11_A ), B=HyperBlock( bounds_f_example11_B), u=util_f_example11, name="example11"),
        Game(A=HyperBlock( bounds_f_example12_A ), B=HyperBlock( bounds_f_example12_B), u=util_f_example12, name="example12"),
        Game(A=HyperBlock( bounds_f_example13_A ), B=HyperBlock( bounds_f_example13_B), u=util_f_example13, name="example13"),
        Game(A=HyperBlock( bounds_f_example14_A ), B=HyperBlock( bounds_f_example14_B), u=util_f_example14, name="example14"),
    ]
    for game in games:
        # solve via Double Oracle
        time_start = time.perf_counter()
        xs, p, ys, q, value_lbs, value_ubs, BRs = double_oracle(game=game, init_algorithm=init_algorithm, maxiter=10, eps=eps)
        time_DO = (time.perf_counter() - time_start)
        DO_outcome = (xs, p, ys, q, value_lbs, value_ubs, BRs)
        # solve via Expected Regret Minimisation
        time_start = time.perf_counter()
        xs, p, ys, q, value_lbs, value_ubs, BRs, as_lens, bs_lens = eps_nash_erm(game=game, init_algorithm=init_algorithm, maxiter=10, eps=eps, C=C)
        time_ERM = (time.perf_counter() - time_start)
        ERM_outcome = (xs, p, ys, q, value_lbs, value_ubs, BRs)
        # plot outcomes
        plot_convergence(DO_outcome[4], DO_outcome[5], DO_outcome[6], ERM_outcome[4], ERM_outcome[5], ERM_outcome[6], game.name, eps, C, time_DO=time_DO, time_ERM=time_ERM)
        plot_complexity(as_lens, bs_lens, BRs, game.name, eps, C)

if __name__ == "__main__":
    #RPS_C_experiment()
    #rosenbrock_experiment()
    convergence_comparison()