import numpy as np

#Rock - Paper - Scissors
# 0       1         2
def utilRPS(x,y):
    if x==y:
        return 0
    else:
        match (x,y):
            case (0,1):
                return -1
            case (0,2):
                return 1
            case (1,0):
                return 1
            case (1,2):
                return -1
            case (2,0):
                return -1
            case (2,1):
                return 1

RPS_actions = np.array([0,1,2])

# x^2 - y^2 + x - y
def util0(x,y):
    return (x**2) - (y**2) +x -y

bounds_0_A = np.array([[-1, 1]])  
bounds_0_B = np.array([[-1, 1]]) 

# 5xy -2x^2 - 2xy^2 -y
def util1(x,y):
    return 5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) - y

bounds_1_A = np.array([[-1, 1]])  
bounds_1_B = np.array([[-1, 1]]) 

# Townsend function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
def util2(x, y):
    term1 = np.power( np.cos( (x-0.1)*y ), 2)
    term2 = x*np.sin( 3*x+y )
    return -term1 -term2

bounds_2_A = np.array([[-2.25, 2.5]])  
bounds_2_B = np.array([[-2.25, 1.75]]) 

# Rosenbrock function from https://en.wikipedia.org/wiki/Test_functions_for_optimization
def util3(x, y):
    return (1-x)**2 + 100*( (y-(x**2)) )**2

bounds_3_A = np.array([[-1.5, 1.5]])  
bounds_3_B = np.array([[-1.5, 1.5]]) 

# ???
def util4(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2

bounds_4_A = np.array([[-1.5, 1.5]])  
bounds_4_B = np.array([[-1.5, 1.5]]) 

#########################################################################################################
# examples following the work of Marika Kosohorska
# https://gitlab.fel.cvut.cz/kosohmar/StayOnTheRidge.jl/-/blob/main/examples/examples.jl


# Example 1 - example from https://proceedings.mlr.press/v195/daskalakis23b/daskalakis23b.pdf (Appendix D)
# solution presented in the paper: (0.5, 0.5)
def util_f_example1(x, y):
     return (x - 0.5) * (y - 0.5)

bounds_f_example1_A = np.array([[0.,1.]])
bounds_f_example1_B = np.array([[0.,1.]])

# Example 2 - normal form game expected utility function (2 players, 2 strategies)
# solution computed by linear programming: (0.92105, 0.92105)
def util_f_example2(x, y):
    A = np.array([[4, -5],
                  [-5, 100]])
    return A[0,0]*x*y \
         + A[0,1]*x*(1-y) \
         + A[1,0]*(1-x)*y \
         + A[1,1]*(1-x)*(1-y)    

bounds_f_example2_A = np.array([[-1.5, 1.5]])  
bounds_f_example2_B = np.array([[-1.5, 1.5]]) 

# Example 3 - function f2 from https://proceedings.mlr.press/v195/daskalakis23b/daskalakis23b.pdf (Appendix E)
# solution presented in the paper: (0,0)
def util_f_example3(x, y):
    left = -x*y -(y**2)/20
    right = 0.1 * util_f_example3_step( (x**2 + y**2)/2) * (y**2)
    return left + right

def util_f_example3_step(z):
    z = 3*(z**2) - 2*(z**3)
    return np.maximum(0, np.minimum(1, z) )

bounds_f_example3_A = np.array([[-1.,1.]])
bounds_f_example3_B = np.array([[-1.,1.]])


# Example 4 - https://link.springer.com/article/10.1007/s10589-019-00141-6 example 5.3 i)
# solution presented in the paper: (0.2540,0.2097,0.2487,0.2944)*10^(-4)
def util_f_example4(x, y): 
    p = x[0]**2 + x[1]**2 -y[0]**2 -y[1]**2
    q = x[0] + y[1] + 1
    return np.array([p/q])

bounds_f_example4_A = np.array([[0.,1.], [0.,1.]])
bounds_f_example4_B = np.array([[0.,1.], [0.,1.]])

# Example 5 - https://arxiv.org/pdf/2109.04178.pdf example 1
# solution presented in the paper: (0.4,0.6)
def util_f_example5(x, y):
    return 2*x*(y**2) - x**2 - y
    
bounds_f_example5_A = np.array([[-1.,1.]])
bounds_f_example5_B = np.array([[-1.,1.]])

# Example 6 - f1 from https://proceedings.mlr.press/v195/daskalakis23b/daskalakis23b.pdf (Appendix E)
# solution presented in the paper: (0,0)
def util_f_example6(x, y):
    inner = (y - 3*x + ((x**3)/20))**2
    base = 4*(x**2) - inner - (y**4)/10
    return base * np.exp(-((x**2) + (y**2))/100)

bounds_f_example6_A = np.array([[-1.,1.]])
bounds_f_example6_B = np.array([[-1.,1.]])


# Example 7 - example 6.3 i) from https://arxiv.org/pdf/1809.01218.pdf
# solution presented in the paper: (-1,-1,1,1,1,1)
def util_f_example7(x, y):
    return np.array([np.sum(x + y) - np.prod(x - y, axis=0)])
    
bounds_f_example7_A = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])
bounds_f_example7_B = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])

# Example 8 - example 6.3 ii) from https://arxiv.org/pdf/1809.01218.pdf
# solution presented in the paper: (-1,1,-1,-1,1,-1)
def util_f_example8(x, y):
    t = y@y - x@x
    for j in range(3):
        for i in range(j):
            t += (x[i]*y[j] - x[j]-y[i])
    return np.array([t])
    
bounds_f_example8_A = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])
bounds_f_example8_B = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])

# Example 9 - function x1^2 - x2^2
# well known solution: (0,0)
def util_f_example9(x, y):
    return x**2 - y**2

bounds_f_example9_A = np.array([[-1.,1.]])
bounds_f_example9_B = np.array([[-1.,1.]])

# Example 10 - monkey saddle
# well known solution: (0,0)
def util_f_example10(x, y):
    return x**3 - 3*x*(y**2)

bounds_f_example10_A = np.array([[-1.,1.]])
bounds_f_example10_B = np.array([[-1.,1.]])