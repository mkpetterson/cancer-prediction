# Standard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from PIL import Image
import scipy

# Tensorflow and Keras
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img




def reshape_image(img):
    """Input img and turn into array """
    
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    return x


def create_new_images(x):
    """Inputs an image and creates more"""
    
    datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            fill_mode='constant',
                            cval=0)  
        
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                         save_to_dir='data/Histology/new_benign',
                         save_prefix='benign',
                         save_format='jpeg'):
        i += 1 
        if i > 3:
            break
            
    return 0