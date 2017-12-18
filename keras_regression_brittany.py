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

# load dataset
dataframe = pandas.read_csv("brittany_numerical.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:26]
Y = dataset[:,26]

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
    model.add(Dense(52, input_dim=26, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=wider_model, epochs=300, batch_size=5, verbose=1)

kfold = KFold(n_splits=20, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
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
    pipeline.fit( X, Y )
#result = pipeline.predict( X[800:,:] )
    result = pipeline.predict( X )
    dataframe['keras'] = result
#numpy.append( dataset, result )
    dataframe.to_csv( "./brittany_learned.csv" )

# Do we have trained network here?
# Is this the evaluation step?
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

