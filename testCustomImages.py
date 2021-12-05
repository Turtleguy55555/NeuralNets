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
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras import layers

x_test_orig = []
x_train_orig = []
x_train = []
y_train = []
x_test = []
y_test = []
friend_orig = []

batch_size = 128
num_classes = 2
epochs = 1

friendY_test = np.array([0,1,1,1,0,0,1,1,1,1,0,0,1,1,0])
friendY_test = keras.utils.to_categorical(friendY_test, num_classes)

p = 50
counter = 0
tc = 0

friendimgsX_test = []
friendimgsY_test = []
for filename in glob.glob('images/*.jpg'):
    im = Image.open(filename)
    im = im.resize((p,p))
    friend_orig.append(np.array(im))
    #im = im.convert("L")
    array = np.asarray(im)
    friendimgsX_test.append(array)

for filename in glob.glob('Test/WithMask/*.png'):


    im = Image.open(filename)

    im = im.resize((p, p))
    arr = np.asarray(im)
    x_test_orig.append(arr)
    #im = im.convert("L")

    array = np.asarray(im)

    features = filters.sobel(im)
    # features = util.invert(features)
    #x_test.append(features)
    x_test.append(array)
    y_test.append(0)
    counter+=1


counter = 0
for filename in glob.glob('Test/WithoutMask/*.png'):

    im = Image.open(filename)

    im = im.resize((p, p))
    arr = np.asarray(im)
    x_test_orig.append(arr)
    #im = im.convert("L")

    array = np.asarray(im)
    # features = util.invert(features)
    #x_test.append(features)
    x_test.append(array)
    y_test.append(1)
    counter+=1

friendimgsX_test = np.array(friendimgsX_test)
friendimgsY_test = np.array(friendimgsY_test)

x_test = np.array(x_test)
yt = []
for i in range(len(y_test)):
    yt.append(y_test[i])

y_test = keras.utils.to_categorical(y_test, num_classes)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(friendimgsX_test, friendY_test, verbose=0)
m = loaded_model.predict(friendimgsX_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
score2 = loaded_model.evaluate(x_test,y_test,verbose=0)
m2 = loaded_model.predict(x_test)
res = []

'''
for i in range(len(m2)):
    if(m2[i][0] > m2[i][1]):
        res.append(0)
    else:
        res.append(1)

badRes = []
print("yt",yt[0])
for i in range(len(res)):
    if(res[i] != yt[i]):
        badRes.append(x_test[i])

print(len(badRes))
for i in range(len(badRes)):
    plt.figure()
    plt.imshow(badRes[i])
    plt.show()
'''

for i in range(len(m)):
    if(m[i][0] > m[i][1]):
        plt.figure()
        plt.title("mask")
        plt.imshow(friend_orig[i])
        plt.show()
    else:
        plt.figure()
        plt.title("no mask")
        plt.imshow(friend_orig[i])
        plt.show()


