#!/usr/bin/python

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy
import pdb

model = Sequential()
#model.add(Dense(10, 64))
model.add(Dense(9, input_dim=9))
model.add(Activation('tanh'))
model.add(Dense(1, input_dim=16))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

dataset = numpy.loadtxt( "brittany_numerical.csv", delimiter=',' )
X = dataset[:,0:9]
Y = dataset[:,26] # brittany

#pdb.set_trace()

X_train = X
Y_train = Y
X_test = X
Y_test = Y

print( "Doing fit." )
model.fit(X_train, Y_train, epochs=20, batch_size=16)

#print( "Doing eval." )
#score = model.evaluate(X_test, Y_test, batch_size=16)

print( "Doing predict" )
score = model.predict(X_test)
print( str( score ) )
