from PIL import Image, ImageFilter, ImageOps
import numpy as np
import glob
import matplotlib.pyplot as plt


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

batch_size = 200
num_classes = 2
epochs = 3

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


for filename in glob.glob('Train/WithMask/*.png'):
    im = Image.open(filename)

    im = im.resize((p, p))
    arr = np.asarray(im)
    x_train_orig.append(arr)
    #im = im.convert("L")

    array = np.asarray(im)
    x_train.append(array)
    y_train.append(0)
    counter+=1
    tc += 1


counter = 0
for filename in glob.glob('Train/WithoutMask/*.png'):
    im = Image.open(filename)

    im = im.resize((p, p))
    arr = np.asarray(im)
    x_train_orig.append(arr)
    #im = im.convert("L")

    array = np.asarray(im)
    x_train.append(array)
    y_train.append(1)
    counter += 1
print(len(x_train))
counter = 0
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
print("counter: ", counter)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test_orig = np.array(x_test_orig)
x_train_orig = np.array(x_train_orig)
print("test shape:", x_test.shape)
friendimgsX_test = np.array(friendimgsX_test)
print("friend test shape:", friendimgsX_test.shape)
print("yt", y_test.shape)

#print("tc:", tc)
'''
plt.figure()
plt.title(y_train[0])
plt.imshow(x_train[0],cmap=plt.cm.gray_r)
plt.figure()
plt.title(y_train[4999])
plt.imshow(x_train[4999], cmap=plt.cm.gray_r)
plt.figure()
plt.title(friendimgsX_test[2])
plt.imshow(friendimgsX_test[2], cmap=plt.cm.gray_r)
plt.show()
'''
yt = []
maskOn = 0
maskOff = 0

for i in range(len(y_test)):
    yt.append(y_test[i])



y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train)



vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(p,p,3))
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation = 'sigmoid'))



model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

m = model.predict(friendimgsX_test)
#y_classes = keras.np_utils.probas_to_classes(m)
#print(y_test)
score1 = model.evaluate(friendimgsX_test,friendY_test,verbose = 0)
print('friend accuracy: ', score1[1])

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
'''
model_json = model.to_json()
with open("modelBest.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("modelBest.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
