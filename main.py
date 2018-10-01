import numpy as np
import matplotlib.pyplot as plt


def importData(filename, dim):
    # Creates a list of datapoints from a tab separated file.
    # All the data is (t, x, y)
    f = open(filename)
    data = ([[] for _ in range(dim)])
    for i in f:
        #Data is tab separated list
        temp = i.split("\t")

        #Remove a trailing newline
        temp[len(temp) - 1] = temp[len(temp) - 1][0:-1]
        
        #Convert to float if possible
        for j in range(dim):
            try:
                data[j].append(float(temp[j]))
            except ValueError:
                continue
    data = np.array(data)
    return data
        
def plot2DData(data):
    plt.plot(data[0], data[1])
    plt.axis([0, max(data[0]), min(data[1]), max(data[1])])
    plt.show()
    
def plot3DData(data):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data[0], data[1])
    plt.subplot(212)
    plt.plot(data[0], data[2])
    plt.show()

def main():
    B = importData("data/video1", 3)
    plot3DData(B)

    return 0

if __name__ == "__main__":
    main()
