"""
The numerical analysis part of this project. Here we will solve our ODE
numerically and plot the results for comparison with the real-world results

"""

import matplotlib.pyplot as plt
import numpy as np

"""
    Interpolates the track and returns the coefficients in decending power
"""
def iptrack(filename):
    data = np.loadtxt(filename, skiprows=2)
    return np.polyfit(data[:1,], data[:,2],15)

"""
    When passed an interpolation and an x value it returns a list of useful 
    values at the point x.
"""
def trvalues(p,x):
    y       = np.polyval(p,x)
    dp      = np.polyder(p)
    dydx    = np.polyval(dp,x)
    ddp     = np.polyder(dp)
    d2ydx2  = np.polyval(ddp, x)
    alpha   = np.arctan(-dydx)
    R       = (1.0 + dydx**2)**1.2 / d2ydx2
    return [y, dydx, d2ydx2, alpha, R]

def a(x, filename):
    p = iptrack(filename)

    return [a_x, a_y]


def main():
    N = 10000
    h = 0.0001

    t_0 = 0
    x_0 = 0.7

    t = np.zeros(N+1)
    x = np.zeros(N+1)
    
    t[0] = t_0
    x[0] = x_0

    t_old = t_0
    x_old = x_0

    
    for n in range(N):
        #TODO: Add the correct formula here to solve the system
        x_new = x_old + h*()
        t[n+1] = t_old+h
        x[n+1] = x_new

        t_old += h
        x_old = x_new

    print r'x_N %f' % x_old

    plt.figure()
    plt.plot(t, x)
    plt.ylabel(r'$h(t)$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.show()



if _name__ == '__main__':
    main()
