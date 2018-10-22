import numpy as np                  #For data analysis
import matplotlib.pyplot as plt     #For basic plotting
from mpl_toolkits import mplot3d    #3D plotting tool for matplotlib
import glob                         #For creating globs from our datapaths

"""
    Global variables contains the filenames, 
    data, regression values and maxima points

    files:
        List of file paths to data files.

    data:
        array of three dimentional vectors with time, x and y points.

    reg:
        array of regression coefficients.
        [a_0, a_1], corresponding to: f(t) = e^(a_1)*e^(a_0t)
    maxima:
        array with arrays of maxima points.
"""
files = []
data = []
reg = []
maxima = []

def importData(filename):
    """
        Using NumPys built in load function it loads the files into NumPy arrays.
        Data format:
        ### Start of file
        mass_A
        't'      'x'       'y'
        t_0     x_0         y_0
        ...
        ### EOF ###
    """
    try:
        data = np.loadtxt(filename, skiprows=2)
    except:
        print "Could not load: ", filename
        return []
    data = data.transpose()

    return data

def findMaxima(time, data):
    """
    By comparing a point x with the next and previous points, 
    we get a crude extrema function
    
    @arg time is a vector of timestamps
    @arg data is a vector of function values at the corresponding index in time.

    returns an array of points[timestamp, maxima_value].
    """
    maxima = []
    for i in range(2, (len(data)-2)):
        if (data[i] > data[i-1]) and (data[i]> data[i+1]):
            if (data[i] > data[i -2]) and (data[i] > data[i+2]):
                maxima.append([time[i],data[i]])
    maxima = np.array(maxima).transpose()
    return maxima


def plot3D(data):
    """
        Creates a 3D plot of the data passed to the function.

        @arg data - This is a 3 dimentional vector on the form [time, x, y]
    """
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
    """
        Creates a 2D plot based on the data passed to the function

        @arg data - This is a 2D array on the form [x, y]
    """
    plt.figure()
    plt.plot(data[0], data[1])
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.show()

def evalPol(reg, time):
    """
        Evaluates the regression with coefficients reg[0] and reg[1] on all points in the array time

        @reg - Has the exponential regression coefficients. f(t) = exp(reg[1]) * exp(reg[0]*t)
        @time - Contains all the points where f should be evaluated.
    """
    #print reg
    pol = []
    for i in time:
        pol.append([i, np.exp(reg[1]) * np.exp(reg[0] * i)])
    pol = np.array(pol).transpose()
    return pol

def plotAll():
    plt.figure(0)
    plt.plot(1)
    for i in range(0, len(maxima)):
        plt.subplot(4, 3, i+1)
        plt.plot(maxima[i][0], maxima[i][1], 'ro', evalPol(reg[i], data[i][0])[0], evalPol(reg[i], data[i][0])[1])
        plt.plot(data[i][0], data[i][2])
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('height')
    plt.show()

def plotDataset(dataset_index):
    plot_title = "Dataset number " + str(dataset_index + 1)
    plt.figure(2)
    set_maxima  = maxima[dataset_index]
    set_pol     = evalPol(reg[dataset_index], data[dataset_index][0])
    raw_data    = data[dataset_index]
    plt.plot(raw_data[0], raw_data[2], linewidth=3.0)
    plt.plot(set_pol[0], set_pol[1], set_maxima[0], set_maxima[1], 'ro', linewidth=4.0)
    plt.xlabel("$time [s]$", fontsize=30)
    plt.ylabel("$height [m]$", fontsize=30)
    plt.title(plot_title)
    plt.grid(True)
    plt.show()

def plotMeanRegression():
    time = data[0][0]
    mean_value = evalPol(reg[0], time)[1]

    for i in range(1, len(reg)):
        mean_value += evalPol(reg[i], time)[1]
    
    plt.figure(3)
    plt.plot(time, mean_value, linewidth=3.0)
    plt.xlabel('$time[s]$', fontsize=24)
    plt.ylabel('$height[m]$', fontsize=24)
    plt.grid(True)
    #plt.title("Mean regression plot")
    plt.show()

def main():
    #Load all datafile names into files
    for i in glob.glob("data/*.data"):
        files.append(i)
        print "File loaded: ", i

    #Import all the data from all our data files.
    for i in files:
        data.append(importData(i))
        print "Data imported from: ", i
    
    #Find all the maxima points of our datasets
    for i in data:
        maxima.append(findMaxima(i[0], i[2]))

    #Do a exponential regression on all the maxima datapoints.
    for i in maxima:
        reg.append(np.polyfit(i[0], np.log(i[1]), 1, w = np.sqrt(i[1])))

    print "Ae^(-at):"
    for i in reg:
        print "A =", np.exp(i[1]),"a =",i[0]
    #plotAll()
    plotDataset(7)
    plotMeanRegression()

    return 0

main()
