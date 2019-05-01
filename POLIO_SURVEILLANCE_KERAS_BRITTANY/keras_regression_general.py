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
import math
import random
import sys
import collections

# Added this to see if theano is configured to use cuda or cpu.
#import theano
#print( theano.config.device )

# load dataset
# the original file I was using had removed some alphanumeric data, partly to get something working quickly.
# the polio...fuller.csv file has 3 additional alpha cols converted to numbers and keeps the year column.

# Other files we've trained on:
# polio_surveillance_numeric_fuller.csv
# brittany_polio_sparse.csv
# DeliveryCosts_KerasReady.csv

# Load data
dataframe = None
dataframe = pandas.read_csv(sys.argv[1], delim_whitespace=False, header=None)
#dataframe = pandas.read_csv(sys.argv[1], delim_whitespace=True, header=None)

dataset = dataframe.values

# Measure data
num_rows = dataset.shape[0]
num_cols = dataset.shape[1]
num_input_cols = num_cols - 1
# split into input (X) and output (Y) variables
# the Y col is the final col. Could find that programmatically instead of hardcoding 30
#X = dataset[:,0:30]
#X = dataset[:,0:7]

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
random.shuffle( dataset )

# X is the full input dataset, Y is the output column
X = dataset[:,0:num_input_cols]
Y = dataset[:,num_input_cols]
# num_input_cols = number of input columns
#num_input_cols = X.shape[1]
# Store the original input dataset so we can use mutate X for our purposes
orig_X = X

# Split the datasets into X Training and Testing. 410 is for Surveillance I believe.
Train_X=X[0:410,]
Train_Y=Y[0:410,]
Test_X=X[411:,]
Test_Y=Y[411:,]

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
    # can set kernel_initializer='normal', not sure if that does anything for us.
    #model.add(Dense(1000, input_dim=num_input_cols, activation='relu'))
    #model.add( Dense(210, input_dim=num_input_cols, kernel_initializer='normal', activation='relu') )
    #model.add( Dense(2000, input_dim=num_input_cols, kernel_initializer='normal', activation='relu') )
    model.add( Dense(2000, input_dim=num_input_cols, kernel_initializer='normal', activation='relu') )
    model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add( Dense(1) )
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def usekfold( num_splits ):
    # evaluate model with standardized dataset
    # the original version here was not using pipeline and was producing bad results.
    # My understanding is that that approach didn't normalize the data. The pipeline, mlp, etc.
    # takes care of unnormalized input columns.
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(( 'mlp', KerasRegressor(build_fn=wider_model, epochs=5000, batch_size=15, verbose=1) ) )
    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=num_splits, random_state=seed, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    with open( "report.txt", "a" ) as report:
    	print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    	report.write("Results: %.2f (%.2f) MSE\n" % (results.mean(), results.std()))
    return -1.0*results.mean()


def doitmyway( X, Y ):
    # evaluate model with standardized dataset 
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    # alternate values: epochs=2500, batch_size=5
    # alternate line for different version of Keras: build_fn=build_fn_reg, hidden_dims=hidden_dims, batch_size=batch_size, nb_epoch=nb_epoch)
    nn_estimator = ('mlp', KerasRegressor(build_fn=wider_model, epochs=5000, batch_size=15, verbose=1))
    estimators.append(nn_estimator)
    pipeline = Pipeline(estimators)
    print( "Let's train on ALL the data (yes, we should really be holding something back for testing. TBD" )
    pipeline.fit( X, Y ) # Should probably use Train_X and Train_Y
    print( "OK, now that we've trained, let's see how well we have learned everything by running all the data through the trained network." )
    result = pipeline.predict( X )
    delta = numpy.subtract( Y, result )
    delta *= delta
    delta = numpy.sqrt( delta )

    # Should probably use Test_X
    if False:
        cum_err = 0
        for i in range(len(Test_Y)):
            print( "Comparing test value {0} with predicted value {1}.".format( Test_Y[i], result[i] ) )
            delta = Test_Y[i]-result[i]
            root_sq_err = math.sqrt( delta * delta )
            cum_err += root_sq_err 
            print( "mean error = " + str( cum_err/(len(X)-410.)) )
    write_predicted_output_to_csv = True
    if write_predicted_output_to_csv:
        print( "Now we'll enter that result into the last column of our dataframe..." )
        dataframe['keras'] = result
        print( "And save it to a file called 'delivery_3cols_learned.csv'" )
        numpy.append( dataset, result )
        dataframe.to_csv( "./delivery_3cols_learned.csv" )
    return sum(delta)/len(delta)

#cols_to_pare = [ 28, 20, 24, 11, 3, 6, 25, 12, 16, 13, 10, 26, 21, 19, 5, 23, 0, 4, 7, 2, 15, 14, 18, 8, 27, 1, 17, 22, 9 ]
#cols_to_pare = [ 2, 1, 10, 13, 16, 14 , 5 , 18 , 6 , 3 , 11 , 8 , 15 , 7 , 4 , 21 , 17 , 20 , 9 , 19 , 0 , 12 ]
#cols_to_pare = [ 5, 11, 3, 19, 2, 16, 14, 10, 15, 17, 13, 1, 9, 4, 21, 8 ] # , 7, 20 ] # adding 20 to end here, I can't believe that's really important
cols_to_pare = [ 9, 8, 21, 13, 16, 11, 4, 18, 14, 2, 15, 1, 5, 17, 10, 12, 3, 20, 0, 19, 7, 6 ]


def remove_zero_cols( my_matrix ):
    col = 0
    while( my_matrix.shape[1] > col ):
        # Find right way to see if matrix/2d-array/dframe width is > col
        sum_of_col = sum(my_matrix[:,col])
        if sum_of_col > 0:
            col += 1
        else:
            my_matrix = numpy.delete( my_matrix, col, 1 )
        
    print( "remove_zero_cols produced dataframe with " + str( my_matrix.shape[1] ) + " columns." ) 
    return my_matrix


prune_column_by_column = False
prune_one_col_at_a_time = False
prune_selected_columns = False

if __name__ == "__main__":
    # "my way" trains on all data and tests aon all, putting 'learned' output as inserted final column. MSE calculation is left as manual
    # step in post.
    #doitmyway( X, Y )
    # K-fold evaluation wasn't working, but now is mostly fixed. Except it's producing 0.3 vs 0.03 that I get manually.
    # Order-of-mag error may seem bad but it's way closer than I was getting.
    usekfold( 10 )

    if prune_selected_columns:
        for x in range( 0,num_input_cols ):
            if x not in cols_to_keep:
                X[:,x] = 0
                print( "Removed col " + str(x) )
                doitmyway()

    # let's iterate over all input columns, remove that column, do full test, and see what difference it makes.
    if prune_one_col_at_a_time:
        rev_map = {}
        # we need a dict that maps hidden_column to mean_error
        hidden_col_2_mean_error = {}
        for x in range( 0,num_input_cols ):
            X = numpy.delete( orig_X, x, axis=1 )
	    with open( "report.txt", "a" ) as report:
		report.write( "Removed col {0}\n".format( x ) )
            print( "Removed col " + str(x) )
            num_input_cols = num_input_cols - 1
            #mean_error = usekfold( 4 )
            mean_error = doitmyway(X, Y) 
            hidden_col_2_mean_error[ x ] = mean_error
        for k in hidden_col_2_mean_error:
                rev_map[ hidden_col_2_mean_error[k] ] = k 
        #rev_map = sorted( collections.OrderedDict( rev_map ).items() ) # looks fancy, but oddly unpredictable and error prone
        print( str( rev_map ) )
        print( "Below is the list of columns (reverse) sorted by the degree to which they seem to be important to predicting the output. First is biggest error ergo most important." )
        reverse_map_keys_sorted = sorted( rev_map.keys(), reverse=True )
        for key in reverse_map_keys_sorted:
            print( rev_map[key] )

    # Below is the code for cumulatively eliminating columns presumably from least to most correlated and seeing how well we can learn
    if prune_column_by_column:
        for x in cols_to_pare[::-1]:
            #X = numpy.delete( X, x, 1 )
            X[:,x] = 0
	    with open( "report.txt", "a" ) as report:
		report.write( "Removed col {0}\n".format( x ) )
            print( "Removed (zeroed) col " + str(x) )
            #X = remove_zero_cols( X )
            usekfold( 5 )
            #doitmyway()

