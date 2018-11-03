"""
The numerical analysis part of this project. Here we will solve our ODE
numerically and plot the results for comparison with the real-world results

"""

import matplotlib.pyplot as plt
import numpy as np
import os

filename = "data/track_pol"
data = np.loadtxt(filename, skiprows=2)
data = data.transpose()
p = np.polyfit(data[1], data[2], 15)
#plt.plot(data[1], np.polyval(p, data[1]))
#plt.show()

def trvalues(x):
    global p
    y       = np.polyval(p,x)
    dp      = np.polyder(p)
    dydx    = np.polyval(dp,x)
    ddp     = np.polyder(dp)
    d2ydx2  = np.polyval(ddp, x)
    alpha   = np.arctan(-dydx)
    R       = (1.0 + dydx**2)**1.2 / d2ydx2
    return [y, dydx, d2ydx2, alpha, R]

def angle(x):
    dp = np.polyder(p)
    alph = np.arctan(-1 * np.polyval(dp, x))
    return alph

def a(x):
    #Baneaksellerasjon
    g = 9.81
    [y, dydx, d2ydx2, alpha, R] = trvalues(x)
    c = 1 + ((2/5)/(0.029*0.011))
    #print "Alpha =", alpha, "angle =", angle(x), "x =", x 
    a = g * np.sin(alpha) / c
    return [a, alpha, y, dydx]


def main():
    N = 200000
    h = 0.0001
    k = 0.3

    t_0 = 0
    y_0 = 0.6
    x_0 = -0.6
    test =  0

    t   = np.zeros(N+1)
    x   = np.zeros(N+1)
    y   = np.zeros(N+1)
    v   = np.zeros(N+1)
    a_b = np.zeros(N+1)
    y_m = np.zeros(N+1)
    v_m = np.zeros(N+1)
    
    
    t[0] = t_0
    y[0] = y_0
    x[0] = x_0
    v[0] = 0

    [a_b[0], alpha, y_m[0], v_m[0]] = a(x_0) 

    for n in range(N):
        t[n+1] = t[n] + h
        
        #Calculate new speeds
        v[n+1] = v[n] + h * a_b[n]

        #Calculate new positions
        x[n+1] = x[n] + h*(v[n] * np.cos(alpha))
        y[n+1] = y[n] - h*(v[n] * np.sin(alpha))
        
        
        #Calculate new acceleration
        [a_b[n+1], alpha, y_m[n+1], v_m[n+1]] = a(x[n+1])
        a_b[n+1] -= v[n] * k
        
        
        print "\033c", "t =", t[n+1]
        print "x =", x[n+1]
        print "y =", y[n+1]
         

        #print "n =", n

    
    plt.figure()
    plt.subplot(221)
    plt.title("Numeric height")
    plt.plot(t, y, linewidth=4.0)
    plt.ylabel(r'$h(t)$ [m]')
    plt.xlabel(r'$t$ [s]')
    plt.grid()

    plt.subplot(222)
    plt.title("Measured height")
    plt.plot(t, y_m, linewidth=4.0)
    plt.ylabel(r'$h(t)$ [m]')
    plt.xlabel(r'$t$ [s]')
    plt.grid()
   
    plt.subplot(223)
    plt.title("Numeric speed")
    plt.plot(t, v, 'g', linewidth=4.0)
    plt.ylabel(r'$v(t)$ [m/s]')
    plt.xlabel(r'$t$ [s]')
    plt.grid()

    plt.subplot(224)
    plt.title("Measured speed")
    plt.plot(t, v_m, 'g', linewidth=4.0)
    plt.ylabel(r'$v(t)$ [m/s]')
    plt.xlabel(r'$t$ [s]')
    plt.grid()

    plt.rcParams.update({'font.size': 30})
    plt.show()



if __name__ == '__main__':
    main()
