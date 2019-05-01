#!/usr/bin/python
"""
Job 1, understand the data.
Spectrograms. About 8000 of them. Though perhaps somehow only 6380?
Each one is 512x128? 1000-1512 is the 512. 128 is the number of frequencies (freq buckets)?
>>> len(df[ df["split"] == "test" ])
800
>>> len(df[ df["split"] == "training" ])
6400
>>> len(df[ df["split"] == "validation" ])
800
>>>
"""

# 3. Import libraries and modules
import sys
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist

import pandas as pd
import pdb

#from matplotlib import pyplot as plt
def GetSpectrograms(df,entropy_loss=True):

    """ Simple loop over spectrograms to create the input stack and return torch.tensor
    classes of the inputs and outputs. 

    The entropy_loss flag specifies the format of the outputs."""

    ## Loop over spectrograms, highlighting missing songs due to corrupted 
    ## files. When the spectrogram is loaded, data is sliced and scaled.
    X = []
    #X = np.empty(0)
    for track_id in df.index:
        fname = "spectrograms/"+str(track_id)+".npz"
        try:
            x = np.load(fname)["log_mel"][:,1000:1512]

            x = x/x.max() # some built-in normalization
        except FileNotFoundError:
            raise FileNotFoundError("Track {} is a corrupted file and must be excluded!".format(track_id))
        if x.shape != (128,512):
            print( "Bad data: wrong shape: {0}.".format( track_id ) )
            continue
        X.append(x)

    #X_arr = np.empty(0,0,0)
    #for x in range(len(X)):
    #    print('.', end='', flush=True)
    #    X_arr = np.append( X_arr, x ) # This is really slow
        #X_arr[x] = X[x]
    print( "\nLoaded {0} npz files/spectrograms.".format( len( X ) ) )
    #X_train is now a simple 3-D python array (list of lists of lists)
    X = np.array( X ).reshape( len( X ), 1, 128, 512 ) 
    #X = np.array( X ).reshape( len( X ), 128, 512 ) 
    # 6380 x 512 x 128
    #X_train = X_train.reshape(X_train.shape[0], 1, 6380, 128)

    Y = df.values # is Y already 1-hot encoded?

    return X, Y

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

def create_spotify_nn():
    model = Sequential()
    model.add(Convolution1D(16, 4, activation='relu', input_shape=(128,512), data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Convolution1D(32, 4, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Convolution1D(64, 4, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))
    return model


def create_neural_network(): 
    # 7. Define model architecture
    
    """
    This is basically the MNIST network which got us to 40% accuracy
    """
    model = Sequential() 
    model.add(Convolution2D(16, (2,2), activation='relu', input_shape=(1,128,512), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32, (2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, (2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Flatten())
    model.add(Dropout(0.9))
    model.add(Dense(80, activation='relu')) # 896 not that great. 128 pretty good. 256->37.7%
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
     
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print( model.summary() )
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def load_data():
    df = pd.read_pickle("response.pkl")

    ## Drop entries associated with corrupted MP3, low resolution files, or shortened tracks. Also, dropping
    ## one experimental song that basically blows up your speakers.
    corrupted = ["29350","29351","99134","108925","133297","17631","17632","17633","17634","17635","107535",
                             "17636","17637","29355","54568","54576","54578","55783","98565","98567","98569","136928"]
    df = df[~df.index.isin(corrupted)]
    return df

def train_and_save(): 
    # 9. Fit model on training data
    #X_train, X_test, Y_train, Y_test = get_data()
    df = load_data()
    ## Get the training and validation sets, randomizing the order of the 
    ## training data
    training_df = df[ ( ( df["split"] == "training" ) | ( df["split"] == "tes" ) ) ].drop("split",axis=1).sample(frac=1)
    #training_df = df[ len(df["split"])>0 ].drop("split",axis=1).sample(frac=0.8)

    ## Get the validation spectrograms. For the training set, this is
    ## done in mini-batches for memory reasons.
    X_train, Y_train = GetSpectrograms(training_df)

    model = create_neural_network()  # only doing this first because troubleshooting setup configurations and want answer fast
    model.fit(X_train, Y_train, batch_size=40, nb_epoch=100, verbose=1)

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

    #X_train, X_test, Y_train, Y_test = get_data()
    df = load_data()
    validation_df = df[df["split"] == "test"].drop("split",axis=1)
    #validation_df = df.drop("split",axis=1)
    X_test, Y_test = GetSpectrograms(validation_df)

    #Predict some data
    # 10. Evaluate model on test data
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print( "testing score from batch test: " + str(score) )

    print( "Doing one-by-one test, displaying error cases." )
    count = 0
    correct = 0
    for datum in X_test:
        #infer = model.predict( datum )
        infer = loaded_model.predict_classes( datum.reshape((1,1,128,512)) )
        #infer = loaded_model.predict_classes( datum.reshape((1,128,512)) )
        if infer[0] != np.argmax(Y_test[count]):
            print( "Error..." + str(infer[0]), str(np.argmax(Y_test[count])) ) 
        else:
            print( "Correct! ({0})".format( np.argmax(Y_test[count]) ) )
            correct += 1
        count += 1
    print( "Accuracy: " + str( float(correct)/count ) )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_and_save()
        load_and_test() 
    elif sys.argv[1] == "--train":
        train_and_save()
    elif sys.argv[1] == "--test":
        load_and_test()
    else:
        print( "Usage: <script> [--train|--test]\nNo arguments means do both." )
