#from prepare import load_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import requests
from io import BytesIO

from PIL import Image

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


# Parameters
input_size = (224,224)

#define input shape
channel = (3,)
input_shape = input_size + channel

#(feature,labels) = load_data()

#x_train,x_test, y_train,y_test = train_test_split(feature,labels,test_size = 0.1) 
categories = ['Black Samurai','Blue Rim','Crown Tail','Cupang Sawah','Halfmoon']

model = tf.keras.models.load_model('d:/Python/BettaFishClassification/model/betafish.h5',compile=False)
#model.evaluate(np.array(x_test),np.array(y_test),verbose = 1)

#prediction =   model.predict(x_test)

# read image
im = Image.open('D:/Python/BettaFishClassification/test1.jpg')
X = preprocess(im,input_size)
X = reshape([X])
y = model.predict(X)

accuracy = str(np.max(y) * 100)
if float(accuracy) > 90:
    print( categories[np.argmax(y)], accuracy )
else:
    print( 'unknown '+categories[np.argmax(y)], accuracy )
#print( categories[np.argmax(y)], np.max(y) )