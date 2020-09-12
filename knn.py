import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter


(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()
image_index = 7777
#print(label_train[image_index])
print(image_test[0])
print(label_test[0])

#plt.imshow(image_train[image_index], cmap = 'Greys')
#plt.show()


# reshape the arrays; flatten out the images
trainingImage = np.reshape(image_train, (60000, 28*28))
testImage = np.reshape(image_test, (10000, 28*28))
print("reshaped")
print(trainingImage.shape)
print(label_train.shape)
print(testImage.shape)
print(label_test.shape)
print("\n\n\n")
#trainingImage = np.swapaxes(trainingImage, 1, 1)
trainingImage = trainingImage[:10000, :]
print(trainingImage.shape)


# get just one image to start with 
test = testImage[0]
label = label_test[0]
prediction = knn(trainingImage, label_train, test, label, 3)
print("correct value " + str(label)



# k nearest neighbours function

def euclideanDistance(train_image, test_image):
   dist = 0
   for i in range(len(train_image)):
      dist += math.pow(train_image[i] - test_image[i], 2)
   euclideanDist = math.sqrt(dist)
   return euclideanDist


def knn(train_set, train_label, test_image, test_label,  k):
   distances_and_labels = []   

   for i in range(len(train_set)):
      dist = euclideanDistance(train_set[i], test_image) 
      distances_and_labels.append((dist, train_label[i]))
   
   sorted_distances_and_labels = sorted(distances_and_labels)
   k_nearest = sorted_distances_and_labels[:k] 
   k_labels = [labels for distances, labels in k_nearest] 
   assigned_label = Counter(k_labels).most_common(1)[0][0]
   return assigned_label

