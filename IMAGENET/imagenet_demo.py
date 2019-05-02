#!/usr/bin/python

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import urllib2
import sys

model = ResNet50(weights='imagenet')

response = urllib2.urlopen( sys.argv[1] )
image_from_web = response.read( response )

#img_path = 'car_battery.jpg'
with open( "/var/tmp/image_to_rec.jpg", "w" ) as tmp_img:
    tmp_img.write( image_from_web )

img_path = "/var/tmp/image_to_rec.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
