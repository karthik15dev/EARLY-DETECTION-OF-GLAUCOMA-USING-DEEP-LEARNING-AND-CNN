import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import csv

# import pydot
#
# from keras import backend as K
#
# import tensorflow as tf
#
from time import time, sleep
#
# from keras.callbacks import TensorBoard

from keras import activations
from matplotlib import pyplot as plt
#
#
# from guided_backprop import GuidedBackprop
# from utils import *
#
# import PIL
#
# from dotenv import load_dotenv


from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam


width = 224
height = 224

model = Sequential()
model.add(Conv2D(64, (4,4), input_shape=(width,height,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights("eighth_try.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])





def save_img(path, savepath, origimg, typeimg, layeridx):

    img = load_img(path, target_size=(224,224))
    x = img_to_array(img) #numpy array
    x = x.reshape(x.shape) #adds on dimension for keras

    model.layers[layeridx].activation = activations.linear
    if typeimg == 'activation':
        img = visualize_activation(model, layeridx, 20, x)

    if typeimg == 'saliency':
        img = visualize_saliency(model, layeridx, 1, x)

    if typeimg == 'cam':
        img = visualize_cam(model, layeridx, 1, x)

    if not os.path.exists('layer-' + savepath):
        os.makedirs('layer-' + savepath)

    if not os.path.exists('image-' + savepath):
        os.makedirs('image-' + savepath)

    combined = str(savepath) + '/' + str(origimg)
    plt.imshow(img)
    plt.savefig('layer-' + combined, dpi=600)
    # plt.imshow(x)
    # plt.savefig('image-' + combined)


types = ['saliency', 'cam']
glauc_imgs = ['G-1-L.jpg', 'G-2-R.jpg']
health_imgs = ['N-1-L.jpg', 'N-2-R.jpg']
for layeridx in [2,3,4,5,6]:
    print("LAYER: " + str(layeridx))
    for typeimg in types:
        print("TYPE: " + str(typeimg))
        for imgidx in range(0,2):
            save_img("rim-flow-datav2/train/glaucoma/" + glauc_imgs[imgidx], 'genimages/' + 'layer_' + str(layeridx) + '/' + typeimg + '/glaucoma', glauc_imgs[imgidx], typeimg, layeridx)
            save_img("rim-flow-datav2/train/healthy/" + health_imgs[imgidx], 'genimages/' + 'layer_' + str(layeridx) + '/' + typeimg + '/healthy', health_imgs[imgidx], typeimg, layeridx)

input_img = load_img("rim-flow-data/train/glaucoma/G-1-L.jpg")
input_img = img_to_array(input_img) #numpy array
input_img = input_img.reshape((1,) + input_img.shape) #adds on dimension for keras
