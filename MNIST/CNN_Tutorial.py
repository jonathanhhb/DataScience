#Original tutorial https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#This link is useful to understand teh concept
#https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/

#Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from random import randint

#Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Check the dimension of training data
print (X_train.shape)
#(60000, 28, 28)

#Preprocess input data
#MNIST images only have a depth of 1, but we must explicitly declare that.
#In other words, we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height).
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

#convert our data type to float32 and normalize our data values to the range [0, 1].
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Define model architecture
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

#Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#Predict some data
result = loaded_model.predict(X_test)

#Visualize ith data

i = randint(0, X_test.shape[0]-1)
plt.imshow(X_train[i])
plt.show()
print ("The prediction of the above image is {0}".format(str(result[i].argmax())))

