import numpy as np

BR_counter = 0

def util0(x,y):
    return (x**2) - (y**2) +x -y

def util1(x,y):
    return 5*x*y - 2*np.power(x, 2) - 2*x*np.power(y,2) - y

def util2(x, y):
    term1 = np.power( np.cos( (x-0.1)*y ), 2)
    term2 = x*np.sin( 3*x+y )
    return -term1 -term2

def util3(x, y):
    return (1-x)**2 + 100*( (y-(x**2)) )**2

def util4(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2

def util_f1(x, y):
    inner = (y - 3*x + ((x**3)/20))**2
    base = 4*(x**2) - inner - (y**4)/10
    return base * np.exp(-((x**2) + (y**2))/100)

bounds_f1_A = np.array([[-1.,1.]])
bounds_f1_B = np.array([[-1.,1.]])

def util_f2(x, y):
    left = -x*y -(y**2)/20
    right = 0.1 * util_f2_S( (x**2 + y**2)/2) * (y**2)
    return left + right

def util_f2_S(z):
    z = 3*(z**2) - 2*(z**3)
    return np.maximum(0, np.minimum(1, z) )

bounds_f2_A = np.array([[-1.,1.]])
bounds_f2_B = np.array([[-1.,1.]])

def util_f_example1(x, y):
     return (x - 0.5) * (y - 0.5)

bounds_f_example1_A = np.array([[0.,1.]])
bounds_f_example1_B = np.array([[0.,1.]])

def util_f_example2(x, y):
    raise NotImplemented #??? where is it?

# x∗ = (0.2540, 0.2097) × 10−4, y∗ = (0.2487, 0.2944) × 10−4
def util_f_example4(x, y): 
    p = x[0]**2 + x[1]**2 -y[0]**2 -y[1]**2
    q = x[0] + y[1] + 1
    return np.array([p/q])

bounds_f_example4_A = np.array([[0.,1.], [0.,1.]])
bounds_f_example4_B = np.array([[0.,1.], [0.,1.]])

def util_f_example5(x, y):
    p = np.sum(x + y)
    for j in range(2):
        for i in range(j):
            p += (x[i]**2 * y[j]**2 - y[i]**2 * x[j]**2)
    q = x[0]**2 + y[1]**2 + x[2]*y[2] + 1
    return np.array([p/q])

bounds_f_example5_A = np.array([[0.,1.], [0.,1.], [0.,1.]])
bounds_f_example5_B = np.array([[0.,1.], [0.,1.], [0.,1.]])


def util_f_example6(x, y):
    return 2*x*(y**2) - x**2 - y
    
bounds_f_example6_A = np.array([[-1.,1.]])
bounds_f_example6_B = np.array([[-1.,1.]])

#x∗ = (−1.0000, −1.0000, 1.0000), y∗ = (1.0000, 1.0000, 1.0000),
#x∗ = (−1.0000, 1.0000, −1.0000), y∗ = (1.0000, 1.0000, 1.0000),
#x∗ = (1.0000, −1.0000, −1.0000), y∗ = (1.0000, 1.0000, 1.0000).
def util_f_example8(x, y):
    return np.array([np.sum(x + y) - np.prod(x - y, axis=0)])
    
bounds_f_example8_A = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])
bounds_f_example8_B = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])

#x∗ = (−1.0000, 1.0000, −1.0000), y∗ = (−1.0000, 1.0000, −1.0000).
def util_f_example9(x, y):
    t = y@y - x@x
    for j in range(3):
        for i in range(j):
            t += (x[i]*y[j] - x[j]-y[i])
    return np.array([t])
    
bounds_f_example9_A = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])
bounds_f_example9_B = np.array([[-1.,1.], [-1.,1.], [-1.,1.]])

#x∗ = (0.3249, 0.3249), y∗ = (1.0000, 0.0000).
def util_f_example10(x, y): 
    #print("util x: ", x)
    #input()
    left = (x[0] + x[1] + y[0] + y[1])**2
    right = 4*(x[0]*x[1] + x[1]*y[0] + y[0]*y[1] + y[1] + x[0])
    return np.array([left + right])

bounds_f_example10_A = np.array([[0.,1.], [0.,1.]])
bounds_f_example10_B = np.array([[0.,1.], [0.,1.]])


def util_f_example11(x, y): 
    return (x-y)**2

bounds_f_example11_A = np.array([[0.,1.]])
bounds_f_example11_B = np.array([[0.,1.]])

def util_f_example12(x, y): 
    return x**2 - 2*x*y + y**2

bounds_f_example12_A = np.array([[-1.,1.]])
bounds_f_example12_B = np.array([[-1.,1.]])
