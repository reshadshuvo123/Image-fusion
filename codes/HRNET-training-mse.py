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

#path='/home/mhoqu001/super-resolution/jj/jj2/jj3/jj6/jj7/jj8/16x'
#os.chdir(path)
#print(os.getcwd())


## low resolution

MatData = h5py.File('Lt1-crop111-hist.mat')
ML=MatData['M1t1']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML1=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML1=np.rollaxis(ML1, 2, 1) # 200, 225, 8 image shape       use for resha
ML1=ML1[1058:1928,3055:3955,0:3]

ML1=ML1.astype('float32')
ML1/= 255
print(ML1.shape)
patchesinput1 = image.extract_patches_2d(ML1, (32, 32))


### High
#### High resolution
MatData = h5py.File('HT1-crop-his.mat')
ML=MatData['h1t1']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML2=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML2=np.rollaxis(ML2, 2, 1) # 200, 225, 8 image shape       use for resha
ML2=ML2[1058:1928,3055:3955,0:3]

ML2=ML2.astype('float32')
ML2/= 255
print(ML2.shape)
patchesinput2 = image.extract_patches_2d(ML2, (32, 32))  # Patch extraction through python build-in function

## Low T2
#Time2 data
#### Low resolution

MatData = h5py.File('Lt2-up-crop-hist.mat')
ML=MatData['l2t2']
ML=np.array(ML)
#ML3=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML3=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML3=np.rollaxis(ML3, 2, 1) # 200, 225, 8 image shape       use for resha
ML3=ML3[1075:1945,3065:3965,0:3]

ML3=ML3.astype('float32')
ML3/= 255
print(ML3.shape)
patchesinput3 = image.extract_patches_2d(ML3, (32, 32))  # Patch extraction through python build-in function

#### High resolution output T2
MatData = h5py.File('HT2-crop.mat')
ML=MatData['h2t2']
ML=np.array(ML)   # I Convert it to array
#ML=ML.reshape(200, 225, 8)  # I reshape it to channel last keras 2 format i.e. height x weight x channel
ML4=np.rollaxis(ML,2, 0) # 200, 8, 225 will be shape       use for reshaping
ML4=np.rollaxis(ML4, 2, 1) # 200, 225, 8 image shape       use for resha
ML4=ML4[1075:1945,3065:3965,0:3]


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
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x





def HRnet3(data1, data2, data3, n_filters=16, dropout=0.1, batchnorm=True):
    ## high resolution
    c1ht1 = conv2d_block(data1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c1lt1 = conv2d_block(data2, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c1lt2 = conv2d_block(data3, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    subtracted1 = keras.layers.Subtract()([c1ht1, c1lt1])
    c1 = keras.layers.add([subtracted1, c1lt2])

    c2 = conv2d_block(c1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c2d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c2)
    ## Mid resolution
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    c22 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c22u = UpSampling2D()(c22)
    c2h = concatenate([c2, c22u])
    c22m = concatenate([c22, c2d])

    c3 = conv2d_block(c2h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c33 = conv2d_block(c22m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    c3d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c3)
    c33u = UpSampling2D()(c33)

    c3h = concatenate([c3, c33u])
    c33m = concatenate([c33, c3d])

    p2 = MaxPooling2D((2, 2))(c33)
    p2 = Dropout(dropout)(p2)

    c3dd = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c3d)
    c444l = concatenate([p2, c3dd])

    c4 = conv2d_block(c3h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c44 = conv2d_block(c33m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c444 = conv2d_block(c444l, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    c4d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c4)
    c4dd = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c4d)
    c44u = UpSampling2D()(c44)
    c44d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c44)
    c444u = UpSampling2D()(c444)
    c444uu = UpSampling2D()(c444u)

    c5h = concatenate([c4, c44u, c444uu])
    c55m = concatenate([c44, c4d, c444u])
    c555l = concatenate([c444, c44d, c4dd])

    c5 = conv2d_block(c5h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c55 = conv2d_block(c55m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c555 = conv2d_block(c555l, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    c555u = UpSampling2D()(c555)
    c555uu = UpSampling2D()(c555u)
    c55u = UpSampling2D()(c55)
    c6h = concatenate([c5, c55u, c555uu])

    c6 = conv2d_block(c6h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    outputs = Conv2D(3, (1, 1), activation='linear')(c6)
    model = Model(inputs=[data1, data2, data3], outputs=[outputs])

    return model

input_L1 = Input(shape=(32, 32, 3), name='Low-inputt1')  # adapt this if using `channels_first` image data format
input_H1 = Input(shape=(32, 32, 3), name='High-inputt1')
input_L2 = Input(shape=(32, 32, 3), name='Low-inputt2')

model = HRnet3(input_L1,input_H1,input_L2, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss= 'mse', metrics=["accuracy"])

model.summary()

train=model.fit({'Low-inputt1': patchesinput1, 'High-inputt1': patchesinput2,'Low-inputt2': patchesinput3}, reference, epochs=35, batch_size=32)

model.save('HRNET-wv2-land-hrnet-hisL2.h5')  #


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
sio.savemat('image-mse-trainig-wv2-land-hisL2-HRNET.mat',{'image': image})

print("************ Test Done ************")


