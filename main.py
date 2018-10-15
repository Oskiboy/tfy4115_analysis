import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import glob

files = []
data = []
reg = []
maxima = []

def importData(filename):
    try:
        data = np.loadtxt(filename, skiprows=2)
    except:
        print "Could not load: ", filename
        return []
    data = data.transpose()

    return data

def findMaxima(time, data):
    maxima = []
    for i in range(2, (len(data)-2)):
        if (data[i] > data[i-1]) and (data[i]> data[i+1]):
            if (data[i] > data[i -2]) and (data[i] > data[i+2]):
                maxima.append([time[i],data[i]])
    maxima = np.array(maxima).transpose()
    return maxima

def plot3D(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    zline = data[2]
    xline = data[1]
    yline = data[0]
    ax.set_zlabel('t')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot3D(xline, yline, zline, 'gray')
    plt.show()

def plot2D(data):
    plt.figure()
    plt.plot(data[0], data[1])
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.show()

def evalPol(reg, time):
    #print reg
    pol = []
    for i in time:
        pol.append([i, np.exp(reg[1]) * np.exp(reg[0] * i)])
    pol = np.array(pol).transpose()
    return pol

def main():
    #Extract all the data filepaths
    for i in glob.glob("data/*.data"):
        files.append(i)

    for i in files:
        #Gathers all the data from all the files.
        data.append(importData(i))
    
    for i in data:
        maxima.append(findMaxima(i[0], i[2]))

    for i in maxima:
        reg.append(np.polyfit(i[0], np.log(i[1]), 1, w = np.sqrt(i[1])))

    a = 0
    b = 0
    for i in reg:
        a += i[0]
        b += i[1]

    super_reg = [a / len(reg), b / len(reg)]
    print "Super regression: ", super_reg
   
    plt.plot(maxima[0][0], maxima[0][1], 'ro', evalPol(reg[0], data[0][0])[0], evalPol(reg[0], data[0][0])[1])
    plt.plot(data[0][0], data[0][2])
    plt.show()

    plt.plot(evalPol(super_reg, data[0][0])[0], evalPol(super_reg, data[0][0])[1])
    plt.show()




    return 0

if __name__ == "__main__":
    main()
