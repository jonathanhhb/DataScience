#!/usr/bin/python

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
#from urllib.request import urlopen
import urllib.request
import random
import sys

model = ResNet50(weights='imagenet')

def downloader(image_url):
    file_name = random.randrange(1,10000)
    full_file_name = "/var/tmp/" + str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)
    return full_file_name

img_path = downloader( sys.argv[1] )
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

