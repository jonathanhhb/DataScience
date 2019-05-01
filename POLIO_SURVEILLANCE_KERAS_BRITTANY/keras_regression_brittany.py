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
import pdb

# Added this to see if theano is configured to use cuda or cpu.
import theano
print( theano.config.device )

# load dataset
# the original file I was using had removed some alphanumeric data, partly to get something working quickly.
# the polio...fullyer.csv file has 3 additional alpha cols converted to numbers and keeps the year column.
dataframe = pandas.read_csv("polio_surveillance_numeric_fuller.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
# the Y col is the final col. Could find that programmatically instead of hardcoding 30
X = dataset[:,0:30]
Y = dataset[:,30]

#X = numpy.delete( X, 4, axis=1 )

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(52, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def usekfold( num_splits ):
    # evaluate model with standardized dataset
    # the original version here was not using pipeline and was producing bad results.
    # My understanding is that that approach didn't normalize the data. The pipeline, mlp, etc.
    # takes care of unnormalized input columns.
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(( 'mlp', KerasRegressor(build_fn=wider_model, epochs=500, batch_size=5, verbose=1) ) )
    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=num_splits, random_state=seed, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def doitmyway():
    # evaluate model with standardized dataset 
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    nn_estimator = ('mlp', KerasRegressor(build_fn=wider_model, epochs=2500, batch_size=5, verbose=1))
    estimators.append(nn_estimator)
    pipeline = Pipeline(estimators)

#X_train = X[0:800,:]
#Y_train = Y[0:800]
    print( "Let's train on ALL the data (yes, we should really be holding something back for testing. TBD" )
    pipeline.fit( X, Y )
#result = pipeline.predict( X[800:,:] )
    print( "OK, now that we've trained, let's see how well we have learned everything by running all the data through the trained network." )
    result = pipeline.predict( X )
    print( "Now we'll enter that result into the last column of our dataframe..." )
    dataframe['keras'] = result
    print( "And save it to a file called 'brittany_learned.csv'" )
#numpy.append( dataset, result )
    dataframe.to_csv( "./brittany_learned.csv" )

# Do we have trained network here?
# Is this the evaluation step?
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

if __name__ == "__main__":
    # "my way" trains on all data and tests aon all, putting 'learned' output as inserted final column. MSE calculation is left as manual
    # step in post.
    #doitmyway()
    # K-fold evaluation wasn't working, but now is mostly fixed. Except it's producing 0.3 vs 0.03 that I get manually.
    # Order-of-mag error may seem bad but it's way closer than I was getting.
    usekfold( 5 )

