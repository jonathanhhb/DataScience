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
#dataframe = pandas.read_csv("polio_surveillance_numeric_fuller.csv", delim_whitespace=False, header=None)
dataframe = pandas.read_csv("brittany_polio_sparse.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
# the Y col is the final col. Could find that programmatically instead of hardcoding 30
#X = dataset[:,0:30]
X = dataset[:,0:7]
num_input_cols = X.shape[1]
orig_X = X
#Y = dataset[:,30]
Y = dataset[:,7]


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
    num_input_cols = X.shape[1]
    #model.add(Dense(52, input_dim=num_input_cols, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(200, input_dim=num_input_cols, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, input_dim=num_input_cols, activation='relu'))
    #model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, activation='relu'))
    #model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(1) )
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
    estimators.append(( 'mlp', KerasRegressor(build_fn=wider_model, epochs=1000, batch_size=5, verbose=1) ) )
    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=num_splits, random_state=seed, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    #pdb.set_trace()
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def doitmyway():
    # evaluate model with standardized dataset 
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    #nn_estimator = ('mlp', KerasRegressor(build_fn=wider_model, epochs=2500, batch_size=5, verbose=1))
    #build_fn=build_fn_reg, hidden_dims=hidden_dims, batch_size=batch_size, nb_epoch=nb_epoch)
    nn_estimator = ('mlp', KerasRegressor(build_fn=wider_model, nb_epoch=2500, batch_size=5))
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

cols_to_pare = [ 28, 20, 24, 11, 3, 6, 25, 12, 16, 13, 10, 26, 21, 19, 5, 23, 0, 4, 7, 2, 15, 14, 18, 8, 27, 1, 17, 22, 9 ]
if __name__ == "__main__":
    # "my way" trains on all data and tests aon all, putting 'learned' output as inserted final column. MSE calculation is left as manual
    # step in post.
    #doitmyway()
    # K-fold evaluation wasn't working, but now is mostly fixed. Except it's producing 0.3 vs 0.03 that I get manually.
    # Order-of-mag error may seem bad but it's way closer than I was getting.
    usekfold( 5 )

    # let's iterate over all input columns, remove that column, do full test, and see what difference it makes.
    #for x in range( 0,29 ):
        #X = numpy.delete( orig_X, x, axis=1 )
        #print( "Removed col " + str(x) )
        #usekfold( 5 )
    if False:
        for x in cols_to_pare:
        #X = numpy.delete( X, x, axis=1 )
            X[:,x] = 0
            print( "Removed (zeroed) col " + str(x) )
            usekfold( 5 )

