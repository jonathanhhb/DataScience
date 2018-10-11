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
	json_file = open('itn_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("itn_model.h5")
	print("Loaded model from disk")
	return loaded_model


scaler = joblib.load( 'dtk_scaler.pkl' )

# Site_Name,x_Temporary_Larval_Habitat,initial_prev,ITN_Coverage

config_json = json.loads( open( "config_amelia.json" ).read() )
input_array = np.zeros( shape=(1,4) )

st = config_json["parameters"]["Site_Type"]
tlh = config_json["parameters"]["x_Temporary_Larval_Habitat"]
ip = config_json["parameters"]["Initial_Prev"]
#itn = config_json["parameters"]["ITN_Coverage"]

input_array[0][0] = st
input_array[0][1] = tlh
input_array[0][2] = ip
#input_array[0][3] = itn 
itn2final = {}

for itn in [ x * 0.01 for x in range(0,100,5)]:
    input_array[0][3] = itn 
    Y_predict = load_model().predict( scaler.transform(input_array) )
    itn2final[ itn ] = Y_predict[0][0]

with open( "data.csv", "w+" ) as file:
    for datum in itn2final:
        file.write( str(datum) + "," + str(itn2final[ datum ]) + "\n" )

#print( itn2final )

