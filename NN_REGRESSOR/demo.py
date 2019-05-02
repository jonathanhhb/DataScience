#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Dense
import numpy
import nnViz
import pdb

seed = 7
numpy.random.seed( seed )

dataset = numpy.loadtxt( "pid.csv", delimiter=',' )
X = dataset[:,0:8]

#X = dataset[:,10:11] # brittany

#X = dataset[:,0:26]
#X = dataset[:,0:8]
#for index in [ 0, 1, 2, 3, 4, 5, 6, 7 ]:
    #training_array = numpy.append( training_array, numpy.take( X, index, 1 ) )
"""
training_array = numpy.vstack( 
        (
            #numpy.take( X, 0, 1 ),
            numpy.take( X, 1, 1 ),
            numpy.take( X, 2, 1 ),
            numpy.take( X, 3, 1 ),
            numpy.take( X, 4, 1 ),
            numpy.take( X, 5, 1 ),
            numpy.take( X, 6, 1 ),
            numpy.take( X, 7, 1 ),
        )
    )
"""
Y = dataset[:,8]
#Y = dataset[:,26] # brittany
#X = training_array.transpose()
#pdb.set_trace()

model = Sequential()
model.add( Dense( 12, input_dim=8, init='uniform', activation='relu' ))
#model.add( Dense( 10, input_dim=1, init='uniform', activation='relu' )) # brittany?
model.add( Dense( 8, init='uniform', activation='relu' ))
#model.add( Dense( 4, init='uniform', activation='relu' ))
model.add( Dense( 1, init='uniform', activation='sigmoid' ))

#nnViz.visualize_model( model )

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

#model.fit( X, Y, nb_epoch=150, batch_size=10 )
model.fit( X, Y, nb_epoch=200, batch_size=10 )


print( "\nEVALUATIONS\n" )
scores=model.evaluate( X, Y )
print( "%s: %.2f%%" % ( model.metrics_names[1], scores[1]*100 ) )
# TRAINING DEMO CODE ENDS HERE

# Now try predictions

#print( "\nPREDICTIONS\n" )
#predictions = model.predict( X )
#rounded = [ round(x) for x in predictions ]
#print( rounded )

