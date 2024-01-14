import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#work in progress

def plot_convergence(LBs_DO, UBs_DO, BRs_DO, LBs_ERM, UBs_ERM, BRs_ERM, name, eps, C, time_DO=None, time_ERM=None, display=False):
    if time_DO is not None:
        label_DO = f"DO, time {time_DO:.4f} s"
        label_ERM = f"ERM, time {time_ERM:.4f} s"
    else:
        label_DO = "DO"
        label_ERM = "ERM"
    plt.plot(BRs_DO, LBs_DO, marker='.', color="blue")
    plt.plot(BRs_DO, UBs_DO, marker='.', color="blue", label=label_DO)
    plt.plot(BRs_ERM,LBs_ERM, marker='.', color="red")
    plt.plot(BRs_ERM,UBs_ERM, marker='.', color="red", label=label_ERM)
    plt.legend()
    plt.xlabel("Best Response calls")
    plt.ylabel("Game value reached")
    plt.title(name+", eps="+str(eps)+", C="+str(C))
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/{name}_brs.png')
    plt.clf()
    
    lens = [len(LBs_DO), len(UBs_DO), len(LBs_ERM), len(UBs_ERM)]
    x_axis = np.arange(max(lens), dtype=int)
    while len(LBs_DO) < len(x_axis):
        LBs_DO.append(LBs_DO[-1])
        UBs_DO.append(UBs_DO[-1])
    while len(LBs_ERM) < len(x_axis):
        LBs_ERM.append(LBs_ERM[-1])
        UBs_ERM.append(UBs_ERM[-1])

    plt.plot(x_axis, LBs_DO, marker='.', color="blue")
    plt.plot(x_axis, UBs_DO, marker='.', color="blue", label=label_DO)
    plt.plot(x_axis, LBs_ERM, marker='.', color="red")
    plt.plot(x_axis, UBs_ERM, marker='.', color="red", label=label_ERM)
    plt.legend()
    plt.title(name+", eps="+str(eps)+", C="+str(C))
    plt.xlabel("Iteration")
    plt.ylabel("Game value reached")
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/{name}_iters.png')
    plt.clf()

def plot_complexity(as_lens, bs_lens, BRs, name, eps, C, display=False):
    n = len(as_lens)
    X = np.arange(0.1, n-1, 0.1)
    nln = np.array([ x*np.log(x) for x in X ])

    # actionspace lens
    max_value = max( max(as_lens), max(bs_lens) )
    coef = abs(np.ceil( max_value/nln[-1] ) + 1)
    scaled_nln = coef*nln

    plt.plot(X, nln, color="black", label="x*ln(x)")
    plt.plot(X, scaled_nln, color="gray", label=str(coef)+"*x*ln(x)")
    plt.plot(as_lens, color="blue", marker='.', label="support of A")
    plt.plot(bs_lens, color="red", marker='.', label="support of B")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Action space sizes per iteration")
    plt.title(name+", eps="+str(eps)+", C="+str(C))
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/{name}_supports.png')
    plt.clf()

    n = len(BRs)
    X = np.arange(0.1, n-1, 0.1)
    nln = np.array([ x*np.log(x) for x in X ])
    coef = abs(np.ceil( BRs[-1]/nln[-1] ) + 1)
    scaled_nln = coef*nln
    plt.plot(X, nln, color="black", label="x*ln(x)")
    plt.plot(X, scaled_nln, color="gray", label=str(coef)+"*x*ln(x)")
    plt.plot(BRs, color="blue", marker='.', label="BestResponse Calls")
    plt.xlabel("Iteration")
    plt.ylabel("Best Response Calls per iteration")
    plt.title(name+", eps="+str(eps)+", C="+str(C))
    plt.legend()
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/{name}_BR_calls.png')
    plt.clf()


def plotCs(Cs, iterations, errors, times, display=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(Cs, errors, marker=".")
    ax1.set_title("Error for chosen C")
    ax1.set_xlabel("value of C")
    ax1.set_ylabel("Error from NE")
    ax2.plot(Cs, iterations, marker=".")
    ax2.set_title("Iterations for chosen C")
    ax2.set_xlabel("value of C")
    ax2.set_ylabel("Iterations")
    ax3.plot(Cs, times, marker=".")
    ax3.set_title("Time[ms] required for chosen C")
    ax3.set_xlabel("value of C")
    ax3.set_ylabel("Time")
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/C_influence.png')
    plt.clf()

def plotCsInfinite(Cs, best_responses, errors, times, display=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(Cs, errors, marker=".")
    ax1.set_title("Error for chosen C")
    ax1.set_xlabel("value of C")
    ax1.set_ylabel("Error from NE")
    ax2.plot(Cs, best_responses, marker=".")
    ax2.set_title("Iterations for chosen C")
    ax2.set_xlabel("value of C")
    ax2.set_ylabel("bestResponse calls")
    ax3.plot(Cs, times, marker=".")
    ax3.set_title("Time[ms] required for chosen C")
    ax3.set_xlabel("value of C")
    ax3.set_ylabel("Time")
    if display:
        plt.show()
    else:
        plt.savefig(f'imgs/C_influence_infinite.png')
    plt.clf()