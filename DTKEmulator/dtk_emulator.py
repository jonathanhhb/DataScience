#!/usr/bin/python
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pdb
import json

# fix random seed for reproducibility
numpy.random.seed(7)

# load json and create model

def load_model():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	return loaded_model


scaler = joblib.load( 'dtk_scaler.pkl' )

config_json = json.loads( open( "config.json" ).read() )
input_array = np.zeros( shape=(1,5) )

bi = config_json["parameters"]["Base_Infectivity"]
bincp = config_json["parameters"]["Base_Incubation_Period"]
binfp = config_json["parameters"]["Base_Infectious_Period"]
abidr = config_json["parameters"]["Acquisition_Blocking_Immunity_Decay_Rate"]
abidbd = config_json["parameters"]["Acquisition_Blocking_Immunity_Duration_Before_Decay"]
input_array[0][0] = bincp
input_array[0][1] = binfp
input_array[0][2] = bi
input_array[0][3] = abidr
input_array[0][4] = abidbd

Y_predict = load_model().predict( scaler.transform(input_array) )
print( Y_predict )

