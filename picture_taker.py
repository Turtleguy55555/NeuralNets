from PIL import Image, ImageFilter, ImageOps
import numpy as np
import glob
import matplotlib.pyplot as plt

from keras.models import model_from_json
from skimage import filters, feature, util
from skimage.color import rgb2gray
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import layers
import cv2

def rescale_frame(frame):
    scale_percent = 100
    width = int(frame.shape[1]*scale_percent/100)
    height = int(frame.shape[0]*scale_percent/100)
    dim = (width,height)
    return cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)

p = 50
vid = cv2.VideoCapture(0)
scale = 30
while(True):
    ret,image = vid.read()

    image = rescale_frame(image)
    height, width, channels = image.shape

    # prepare the crop
    centerX, centerY = int(height /2), int(width / 2)
    radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = image[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    cv2.imshow('my webcam', resized_cropped)

    '''
    image = rescale_frame(image)
    resized_cropped = image
    cv2.imshow('my webcam', resized_cropped)
    '''

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()
imager = []
cv2.imwrite("image.jpg",resized_cropped)
img = Image.open("image.jpg")
img = img.resize((p, p))
img = np.asarray(img)
imager.append(img)

imager = np.array(imager)



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

m = loaded_model.predict(imager)
print(m)
if(m[0][0] > m[0][1]):
    plt.figure()
    plt.title("mask")
    plt.imshow(img)
    plt.show()
else:
    plt.figure()
    plt.title("no mask")
    plt.imshow(img)
    plt.show()



