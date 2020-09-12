import tensorflow as tf
import numpy as np
import math
from keras.utils import np_utils
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y, epochs):
        self.input = x
        self.output = y
        self.iters = epochs
        self.errors = np.zeros(epochs)
        self.learning_rate = 0.5
        input_dim = x.shape[1]          # trainingImages - 784
        output_dim = y.shape[1]         # trainingLabels - 10
        num_neurons = 128               # arbitrary number of neurons

        # weights and biases for the input and hidden layers
        self.weight1 = np.random.randn(input_dim, num_neurons)
        self.weight2 = np.random.randn(num_neurons, output_dim)
        self.bias1 = np.zeros((1, num_neurons))
        self.bias2 = np.zeros((1, output_dim))
    
    def feed_forward(self):
        # compute output of the first layer
        z1 = np.dot(self.input, self.weight1) + self.bias1
        print(z1.shape)
        self.a1 = 1 / (1 + np.exp(-z1))

        # compute output of the hidden layer
        z2 = np.dot(self.a1, self.weight2) + self.bias2
        self.a2 = softmax(z2)
    
    def back_propagate(self, counter):
        loss = error(self.a2, self.output)
        print("Error: ", loss)
        self.errors[counter] = loss

        delta_a2 = cross_entropy(self.a2, self.output)
        self.weight2 -= self.learning_rate * np.dot(self.a1.T, delta_a2)
        self.bias2 -= self.learning_rate * np.sum(delta_a2, axis = 0, keepdims = True)

        delta_z1 = np.dot(delta_a2, self.weight2.T)
        delta_a1 = delta_z1 * d_sigmoid(self.a1)
        self.weight1 -= self.learning_rate * np.dot(self.input.T, delta_a1)
        self.bias1 -= self.learning_rate * np.sum(delta_a1, axis = 0)

    def predict(self, data):
        self.input = data
        self.feed_forward()
        return self.a2.argmax()

def softmax(z):
    exps = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exps / np.sum(exps, axis = 1, keepdims = True)

def d_sigmoid(z):
    return z * (1 - z)

def error(prediction, true):
    num_samples = true.shape[0]
    log_p = -np.log(prediction[np.arange(num_samples), true.argmax(axis = 1)])
    loss = np.sum(log_p) / num_samples
    return loss
       
def cross_entropy(prediction, true):
    num_samples = true.shape[0]
    return (prediction - true) / num_samples

def get_accuracy(x, y):
    accuracy = 0
    for xx,yy in zip(x, y):
        s = NeuralNet.predict(xx)
        if s == np.argmax(yy):
            accuracy +=1
    return accuracy / len(x)*100

def main():
    # import MNIST dataset and roll out the images
    (trainingImages, trainingLabels), (testingImages, testingLabels) = tf.keras.datasets.mnist.load_data()

    trainingImages = np.reshape(trainingImages, (60000, 784)).astype(np.float32)
    testingImages = np.reshape(testingImages, (10000, 784)).astype(np.float32)
    print(trainingLabels.shape)
    
    trainingLabels = np_utils.to_categorical(trainingLabels) 
    testingLabels = np_utils.to_categorical(testingLabels)

    # scale trainingImages between 0.01 and 1
    trainingImagesScaled = trainingImages * (0.99 / 255) + 0.01
    testingImagesScaled = testingImages * (0.99 / 255) + 0.01
    print(trainingImagesScaled.shape)
    
    
    num_epoch = 1000
    NeuralNet = NeuralNetwork(trainingImagesScaled, trainingLabels, num_epoch)

    for i in range(num_epoch):
        NeuralNet.feed_forward()
        NeuralNet.back_propagate(i)

	
    print("Training accuracy: ", get_accuracy(trainingImagesScaled, trainingLabels))
    print("Test accuracy: ", get_accuracy(testingImagesScaled, testingLabels))

    x_axis = np.arange(1, num_epoch + 1)

    plt.plot(x_axis, NeuralNet.errors)
    plt.title("Error over epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()

if __name__ == "__main__":
    main()
    
