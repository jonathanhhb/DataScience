#!/usr/bin/python

"""
Script to generate a table of data where output is some function of some inputs to do sanity test of random forest solution.
"""
import random

def func( a,b,c,d,e ):
    return a+2*b+c+e

mymax = 1000
#with open( "generated_data.csv", "w" ) as gen_d:
print( "A,B,C,D,E,Y" )
for x in range(0,mymax):
    a = x
    b = x
    b = random.randint(0,mymax)
    c = random.randint(0,mymax)
    d = random.randint(0,mymax)
    e = random.randint(0,mymax)

    y = func( a,b,c,d,e )
    print( "{0},{1},{2},{3},{4},{5}".format( a, b, c, d, e, y ) )
    #gen_d.write( "%d,%d,%d,%d,%d,%d\n", x, 10, random.randint(0,1000), random.randint(0,1000), random.randint(0,1000), y )

