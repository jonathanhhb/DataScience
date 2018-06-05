#!/usr/bin/python

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pdb
import math
import random
import sys

# load dataset
input_data = "dtk_generic_08_6params5values.csv"
if len(sys.argv) > 1:
    input_data = sys.argv[1]
dataframe = pandas.read_csv(input_data, delim_whitespace=False, header=None)
dataset = dataframe.values

random.shuffle( dataset )
X = dataset[:,1:6]
num_input_cols = X.shape[1]
orig_X = X
Y = dataset[:,7]

Train_X=X
Train_Y=Y
Test_X=X
Test_Y=Y
model = None
pipeline = None
scaler = None

# define wider model
def wider_model():
    # create model
    global model
    model = Sequential()
    num_input_cols = X.shape[1]
    model.add(Dense(20, input_dim=num_input_cols, activation='relu', kernel_initializer='uniform' ))
    model.add(Dense(1) )
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 120
numpy.random.seed(seed)

def usekfold( num_splits ):
    # evaluate model with standardized dataset
    # the original version here was not using pipeline and was producing bad results.
    # My understanding is that that approach didn't normalize the data. The pipeline, mlp, etc.
    # takes care of unnormalized input columns.
    estimators = []
    global scaler
    scaler = StandardScaler()
    estimators.append(('standardize', scaler ))
    estimators.append(( 'mlp', KerasRegressor(build_fn=wider_model, epochs=20, batch_size=100, verbose=1) ) )
    global pipeline
    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=num_splits, random_state=seed, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def just_train():
    # evaluate model with standardized dataset 
    estimators = []
    global scaler
    scaler = StandardScaler()
    estimators.append(('standardize', scaler ))
    nn_estimator = ('mlp', KerasRegressor(build_fn=wider_model, epochs=20, batch_size=100))
    estimators.append(nn_estimator)
    global pipeline
    pipeline = Pipeline(estimators)

    print( "Let's train on ALL the data (yes, we should really be holding something back for testing. TBD" )
    pipeline.fit( Train_X, Train_Y )

    print( "OK, now that we've trained, let's see how well we have learned everything by running all the data through the trained network." )
    input_array = numpy.zeros( shape=(1,5) )
    input_array[0][0] = 0
    input_array[0][1] = 24
    input_array[0][2] = 0.048
    input_array[0][3] = 0.1
    input_array[0][4] = 60
    #pdb.set_trace()
    regout = model.predict( input_array )[0][0]
    print( "Output of trained model with input from scenario values: " + str(regout) )
    result = pipeline.predict( Test_X )

    cum_err = 0
    for i in range(len(Test_Y)):
        print( "Comparing test value {0} with predicted value {1}.".format( Test_Y[i], result[i] ) )
        delta = Test_Y[i]-result[i]
        root_sq_err = math.sqrt( delta * delta )
        cum_err += root_sq_err 
        print( "mean error = " + str( cum_err/(len(X))) )

if __name__ == "__main__":
    # "my way" trains on all data and tests aon all, putting 'learned' output as inserted final column. MSE calculation is left as manual
    # step in post.
    just_train()
    # K-fold evaluation wasn't working, but now is mostly fixed. Except it's producing 0.3 vs 0.03 that I get manually.
    # Order-of-mag error may seem bad but it's way closer than I was getting.
    #usekfold( 5 )

    # Now that we're done training, let's save this entire model so we can use it as an emulator later
    model_json = model.to_json()
    with open( "model.json", "w" ) as json_file:
        json_file.write( model_json )
    model.save_weights( "model.h5" )
    # just copying code from s/o here. 

    # Next: Pickle the 'standardizer', not the entire pipeline.
    joblib.dump( scaler, 'dtk_scaler.pkl' ) 

