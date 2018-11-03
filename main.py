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

def plotDataset(dataset_index):
    """
    This plot acts up
    """
    plot_title = "Dataset number " + str(dataset_index + 1)
    my_fig = plt.figure(2)
    
    set_maxima  = maxima[dataset_index]
    set_pol     = evalPol(reg[dataset_index], data[dataset_index][0])
    raw_data    = data[dataset_index]

    plt.plot(raw_data[0], raw_data[2], 'g',linewidth=1.5)
    plt.plot(set_pol[0], set_pol[1], set_maxima[0], set_maxima[1], 'ro', linewidth=4.0)
    plt.xlabel("time [s]")
    plt.ylabel("height [m]")
    plt.title(plot_title)
    plt.grid(True)
    plt.rcParams.update({'font.size': 30}) 
    #plt.show()
    my_fig.savefig('dataset.eps', format='eps')

def plotMeanRegression():
    time = data[0][0]
    mean_value = evalPol(reg[0], time)[1]

    for i in range(1, len(reg)):
        mean_value += evalPol(reg[i], time)[1]
    
    fig = plt.figure(3)
    plt.plot(time, mean_value, linewidth=4.0)
    plt.xlabel('time [s]', fontsize=30)
    plt.ylabel('height [m]', fontsize=30)
    plt.grid(True)
    #plt.title("Mean regression plot")
    plt.rcParams.update({'font.size': 30})
    fig.tight_layout() 
    #plt.show()
    fig.savefig('mean_regression.eps', format='eps')

def main():
    #Load all datafile names into files
    for i in glob.glob("data/*.data"):
        files.append(i)
        #print "File loaded: ", i

    #Import all the data from all our data files.
    for i in files:
        data.append(importData(i))
        #print "Data imported from: ", i
    
    #Find all the maxima points of our datasets
    for i in data:
        maxima.append(findMaxima(i[0], i[2]))

    #Do a exponential regression on all the maxima datapoints.
    for i in maxima:
        reg.append(np.polyfit(i[0], np.log(i[1]), 1, w = np.sqrt(i[1])))
    """
    print "Ae^(-at):"
    for i in reg:
        print "A =", np.exp(i[1]),"a =",i[0]
    """
    #plotAll()
    plotDataset(7)
    plotMeanRegression()
    plt.show()

    return 0

main()
