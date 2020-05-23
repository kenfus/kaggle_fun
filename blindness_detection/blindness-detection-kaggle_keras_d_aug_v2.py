# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import transform
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
from IPython.display import Image
from keras.preprocessing import image
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers, initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, ELU, PReLU
from keras import regularizers
from sklearn.utils import class_weight
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# example of loading an image with the Keras API

from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage.io import imread,imshow
from skimage.util import crop
from scipy import ndimage
from PIL import *
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import wandb
from wandb.keras import WandbCallback
wandb.init("blindness-detection-diabetes-kaggle") # Initializes wandb


#Try to make it use all CPUs
import tensorflow as tf
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))

print("Loading succesful")
print(os.listdir("input/"))
print("With batchnorm and more data-augmentation")


def resize_to_square(img):
    shape = img.shape
    min_shape = min(shape[0], shape[1])
    x1, x2 = int(shape[0]/2-min_shape/2),int(shape[0]/2+min_shape/2)
    y1, y2 = int(shape[1]/2-min_shape/2),int(shape[1]/2+min_shape/2)
    img = img[x1:x2, y1:y2]
    return img

# load the image
img_healthy = imread('input/train_images/002c21358ce6.png')
resize_x, resize_y = 339, 339

def resize(img):
    image_resized = transform.resize(img, (resize_x, resize_y), anti_aliasing=False, mode='constant')
    return image_resized
size_flat = img_healthy.flatten().shape

train_csv = pd.read_csv('input/train.csv')
train_csv['id_code'] = train_csv['id_code']+'.png'

train_images_list = os.listdir("input/train_images")
list_len = len(train_images_list)
print(list_len)


images = []
labels = []
loop_nr = 1


for image in train_images_list:
    img = imread('input/train_images/'+str(image))
    img = resize_to_square(img)
    img = resize(img)
    images.append(img)
    label = train_csv.loc[train_csv['id_code'] == str(image), 'diagnosis'].iloc[0]
    if loop_nr%100 < 5:
        print(loop_nr, " of ", list_len, "Label: ", label)
    labels.append(label)
    loop_nr += 1

images = np.asarray(images, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int8)



n_values = np.max(labels) + 1
labels = np.eye(n_values)[labels]
labels_multi = np.empty(labels.shape, dtype=labels.dtype)
labels_multi[:, 4] = labels[:, 4]


for i in range(3, -1, -1):
    labels_multi[:, i] = np.logical_or(labels[:, i], labels_multi[:, i+1])


x_train, x_val, y_train, y_val = train_test_split(
    images, labels_multi,
    test_size=0.15,
    random_state=2019
)


print(labels_multi[:5])

BATCH_SIZE = 256
shift = 0.0
rotation_range = 180
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=rotation_range,
        zoom_range=0.15, horizontal_flip=True,
        vertical_flip=True,fill_mode='constant', cval=0.)
datagen.fit(x_train)

data_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
#images_res = np.reshape(images,(-1,resize_x, resize_y,3))
#labels_res = np.reshape(labels, (-1,1))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1

        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model_v2.h5')

        return

kappa_metrics = Metrics()

initializer ='glorot_normal'
reg= 'l2'

def get_model():
    K.clear_session()
    model = Sequential()
    model.add(Conv2D(192,(7,7), input_shape = (resize_x,resize_y,3), strides = 2, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,3), padding='SAME', strides = 2))


    model.add(Conv2D(256,(3,3), padding='SAME', strides = 1, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,3), padding='SAME', strides = 2))

    model.add(Conv2D(384,(3,3), padding='SAME', strides = 1, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,3), padding='SAME', strides = 2))

    model.add(Conv2D(496,(3,3), padding='SAME', strides = 1, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(496,(3,3), padding='SAME', strides = 1, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,3), padding='SAME', strides = 2))

    model.add(Conv2D(384,(3,3), padding='SAME', strides = 1, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,3), padding='SAME', strides = 2))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(4096, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(2048, kernel_initializer=initializer, kernel_regularizer=reg))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(5,activation = 'sigmoid', kernel_initializer=initializer, kernel_regularizer=reg))

    model.summary()
    return model

model = get_model()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00005)

model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), metrics = ['accuracy'])

wandb_callback = WandbCallback(generator = data_generator, data_type="image", save_model = False)

history = model.fit_generator(data_generator, epochs = 300, validation_data = [x_val, y_val], steps_per_epoch=4*images.shape[0] / BATCH_SIZE,
            callbacks=[wandb_callback, kappa_metrics,reduce_lr], shuffle = True, verbose = 2)
