import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
import math
from scipy.stats import pearsonr


def randomMatrix(m, n):
    # generate a random mxn matrix
    A = np.random.rand(m, n)
    return A
        
# plot the test signals        
def plotTestSignals(sounds, status):
    X = np.arange(1, 41).tolist()
    sound1 = sounds[0, :]
    sound2 = sounds[1, :]
 #   sound3 = sounds[2, :]

    # create figure and plots
    fig, axs = plt.subplots(2)
    if (status == 0):
        fig.suptitle('Original signals')
    elif (status == 1):
        fig.suptitle('Mixed signals')
    elif (status == 2):
        fig.suptitle('Recovered signals')
    axs[0].plot(X, sound1, 'b')
    axs[1].plot(X, sound2, 'g')
 #   axs[2].plot(X, sound3, 'r')
    plt.show()


# implement the ICA algorithm to unmix the signals
def ICA(X, n, m, t, rMax, learnRate):
    # generate W with small random variables
    W = np.random.rand(m, n) * (0.1-0)
                                            
    # gradient descent implementation that ends upon convergence or rMax iterations
    for i in range(rMax):
        # compute current estimate Y
        Y = np.matmul(W, X) 
        Z = computeZ(Y, n, t)

        # This might be the problem?
        deltaW = learnRate * np.matmul((np.identity(n) + np.matmul((1 - 2 * Z), np.transpose(Y))), W)
        W += deltaW
        if (i % 100 == 0):
            print(i)

    return W
        
# compute the Z value
def computeZ(Y, n, t):
    Z = np.ones((n, t))
    for i in range(n):
        for j in range(t):
            Z[i, j] = 1 / (1 + math.exp(-1 * Y[i, j]))
    return Z

def main():
    # load test arrays and plot the original signals
    test = loadmat('./icaTest.mat')
    testA = test['A']
    testU = test['U']
    testX = np.matmul(testA, testU)
    
    
    U = loadmat('./sounds.mat')['sounds']
    U = U[:3, :]
    
    # generate A
    A = randomMatrix(3, 3)

    # mix the signals
    X = np.matmul(A, U)
    
    # carry out ICA
    W = ICA(X, 3, 3, 44000, 10000, 0.001)
    recovered = np.matmul(W, X)
    print(recovered.shape)
    for i in range(3):
        recovered[i, :] = (recovered[i, :] - np.min(recovered[i, :]))/np.ptp(recovered[i, :])

    fig, axs = plt.subplots(6, 1)
    X = np.arange(1, 44001).tolist()
    axs[0].plot(X, U[0, :], 'b')
    axs[1].plot(X, U[1, :], 'g')
    axs[2].plot(X, U[2, :], 'r')
    axs[3].plot(X, recovered[0, :], 'b')
    axs[4].plot(X, recovered[1, :], 'g')
    axs[5].plot(X, recovered[2, :], 'r')
    plt.show()

    
    # mix the signals and print them
 #   testX = np.matmul(testA, testU)
    

    # carry out ICA and print the recovered signals
 #   W = ICA(testX, 3, 3, 40, 100000, 0.01)
    
 #   recovered = np.matmul(W, testX)
 #   print(recovered.shape)
 #   for i in range(3):
 #       recovered[i, :] = (recovered[i, :] - np.min(recovered[i, :]))/np.ptp(recovered[i, :])
        
 #   fig, axs = plt.subplots(6, 1)
 #   X = np.arange(1, 41).tolist()
 #   axs[0].plot(X, testU[0, :], 'b')
 #   axs[1].plot(X, testU[1, :], 'g')
 #   axs[2].plot(X, testU[2, :], 'r')
 #   axs[3].plot(X, recovered[0, :], 'b')
 #   axs[4].plot(X, recovered[1, :], 'g')
 #   axs[5].plot(X, recovered[2, :], 'r')
 #  plt.show()
 #   
 #   plotTestSignals(testU, 0)
 #   plotTestSignals(recovered, 2)
     

 #   compare all of the originals with all of the recovered
    accuracies = np.ones((3, 3))

    for i in range(3):
        for j in range(3):
             coeff = pearsonr(U[i], recovered[j])
             accuracies[i][j] = coeff[0]
    print(accuracies)

    

if __name__ == "__main__":
    main()

    
