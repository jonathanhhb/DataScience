#!/usr/bin/python

# 3. Import libraries and modules
import sys
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from matplotlib import pyplot as plt

def get_data(): 
    # 4. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
     
    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
     
    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, X_test, Y_train, Y_test

def create_neural_network(): 
    # 7. Define model architecture
    model = Sequential()
     
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
    model.add(Convolution2D(32, (3, 3), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
     
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))
    model.add(Dense(10, activation='sigmoid'))
     
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_and_save(): 
    # 9. Fit model on training data
    model = create_neural_network() 
    X_train, X_test, Y_train, Y_test = get_data()
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=5, verbose=1)
    #Save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_and_test():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("Loaded model from disk")

    X_train, X_test, Y_train, Y_test = get_data()

    #Predict some data
    # 10. Evaluate model on test data
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print( "testing score from batch test: " + str(score) )

    #pdb.set_trace()
    print( "Doing one-by-one test, displaying error cases." )
    count = 0
    for datum in X_test:
        #infer = model.predict( datum )
        infer = loaded_model.predict_classes( datum.reshape((1,1,28,28)) )
        if infer[0] != np.argmax(Y_test[count]):
            print( str(infer), str(np.argmax(Y_test[count])) ) 
        count += 1

if __name__ == "__main__":
    if len(sys.argv) == 0:
        train_and_save()
        load_and_test() 
    elif sys.argv[1] == "--train":
        train_and_save()
    elif sys.argv[1] == "--test":
        load_and_test()
    else:
        print( "Usage: <script> [--train|--test]\nNo arguments means do both." )
