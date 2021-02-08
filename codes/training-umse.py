import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

#from tqdm import tqdm_notebook, tnrange
from itertools import chain
#from skimage.io import imread, imshow, concatenate_images
#from skimage.transform import resize
#from skimage.morphology import label
#from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_im
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, merge

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
from keras.utils import np_utils
import os.path
import h5py
import keras
import time
import numpy as np
from scipy import misc
import sklearn
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from keras.models import load_model
import numpy
import sklearn
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from keras.models import load_model
import numpy
import sklearn

#import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D,BatchNormalization
#import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
import scipy.io as sio

#dimension = 33

path='/home/mhoqu001/super-resolution/jj/jj2/jj3/jj6/jj7/jj8/land-sen/land-sen/mse'
os.chdir(path)
print(os.getcwd())


## low resolution

MatData = h5py.File('Low1.mat')
ML=MatData['L1']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML1=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML1=np.rollaxis(ML1, 2, 1) # 200, 225, 8 image shape       use for resha
#ML1=ML1[1000:2024,3000:4024,0:3]

ML1=ML1.astype('float32')
ML1/= 255
print(ML1.shape)
patchesinput1 = image.extract_patches_2d(ML1, (32, 32))


### High
#### High resolution
MatData = h5py.File('High1.mat')
ML=MatData['H1']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML2=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML2=np.rollaxis(ML2, 2, 1) # 200, 225, 8 image shape       use for resha
#ML2=ML2[1000:2024,3000:4024,0:3]

ML2=ML2.astype('float32')
ML2/= 255
print(ML2.shape)
patchesinput2 = image.extract_patches_2d(ML2, (32, 32))  # Patch extraction through python build-in function

## Low T2
#Time2 data
#### Low resolution

MatData = h5py.File('Low2.mat')
ML=MatData['L2']
ML=np.array(ML)
#ML3=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML3=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML3=np.rollaxis(ML3, 2, 1) # 200, 225, 8 image shape       use for resha
#ML3=ML3[1000:2024,3000:4024,0:3]

ML3=ML3.astype('float32')
ML3/= 255
print(ML3.shape)
patchesinput3 = image.extract_patches_2d(ML3, (32, 32))  # Patch extraction through python build-in function

#### High resolution output T2
MatData = h5py.File('High2.mat')
ML=MatData['H2']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML4=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML4=np.rollaxis(ML4, 2, 1) # 200, 225, 8 image shape       use for resha
#ML4=ML4[1000:2024,3000:4024,0:3]


ML4=ML4.astype('float32')
ML4/= 255
print(ML4.shape)
reference = image.extract_patches_2d(ML4, (32, 32))  # Patch extraction through python build-in function


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),  kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def U_encoder(data, n_filters=16, dropout=0.1, batchnorm=True):

    """Function to define the UNET Model"""
    # Contracting Path t1 Low
    c1 = conv2d_block(data, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    return c1, c2, c3, c4, c5

'''
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam


class VGG_LOSS(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        #vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19 = VGG19('~/.keras/models/*')
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model1 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model1.trainable = False

        return K.mean(K.square(model1(y_true) - model1(y_pred)))
image_shape=(32,32,3)
loss= VGG_LOSS(image_shape)
'''
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(y_true, y_pred):
    img_nrows = img_ncols= 32
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def get_unet(data1,data2,data3, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""

    c1, c2, c3, c4, c5 = U_encoder(data1, n_filters=16, dropout=0.1, batchnorm=True)
    c11, c22, c33, c44, c55 = U_encoder(data2, n_filters=16, dropout=0.1, batchnorm=True)
    c111, c222, c333, c444, c555 = U_encoder(data3, n_filters=16, dropout=0.1, batchnorm=True)

    subtracted5 = keras.layers.Subtract()([c55, c5])
    merge5 = keras.layers.add([subtracted5,c555])
    subtracted4 = keras.layers.Subtract()([c44, c4])
    merge4 = keras.layers.add([subtracted4,c444])
    subtracted3 = keras.layers.Subtract()([c33, c3])
    merge3 = keras.layers.add([subtracted3,c333])
    subtracted2 = keras.layers.Subtract()([c22, c2])
    merge2 = keras.layers.add([subtracted2,c222])
    subtracted1 = keras.layers.Subtract()([c11, c1])
    merge1 = keras.layers.add([subtracted1,c111])

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(merge5)
    u6 = concatenate([u6, merge4, c444])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, merge3, c333])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, merge2, c222])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, merge1, c111])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(3, (1, 1), activation='linear')(c9)
    model = Model(inputs=[data1,data2,data3], outputs=[outputs])
    return model


input_L1 = Input(shape=(32, 32, 3), name='Low-inputt1')  # adapt this if using `channels_first` image data format
input_H1 = Input(shape=(32, 32, 3), name='High-inputt1')
input_L2 = Input(shape=(32, 32, 3), name='Low-inputt2')

model = get_unet(input_L1,input_H1,input_L2, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss= 'mse', metrics=["accuracy"])

model.summary()

train=model.fit({'Low-inputt1': patchesinput1, 'High-inputt1': patchesinput2,'Low-inputt2': patchesinput3}, reference, epochs=35, batch_size=32)

model.save('model-onelyr-shared-hisL2.h5')  #


result=model.predict([patchesinput1,patchesinput2,patchesinput3])

import sklearn
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
from keras.models import load_model
import numpy
import sklearn


size=ML3.shape
image=sklearn.feature_extraction.image.reconstruct_from_patches_2d(result, size)

import scipy.io as sio
sio.savemat('image-mse-trainig-hisL2.mat',{'image': image})

print("************ Test Done ************")

