# Standard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from PIL import Image
import scipy

# Fast AI model
from fastai.vision import *
from fastai.metrics import error_rate

# Xception/Tensorflow
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img





class FastAI():
    
    def __init__(self, path):
        self.path = path
        self.classes = sorted([d for d in os.listdir(path)])        
        
    def verify_images(max_size):
        """Verify images in folder and reduce if any dimension is above max_size"""
                
        # Verify images and set max size
        for folder in ['benign', 'malignant']:
            print(folder)
            verify_images(path/folder, delete=True, max_size=max_size)
            
        return "done"   
    
        
    def fit(self, model_name):
        """ Fits model"""
        
        # Set up image data bunch
        np.random.seed(np.random.randint(0,100))
        data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,
                                 ds_tfms=get_transforms(), size=224, 
                                  num_workers=4).normalize(imagenet_stats)
        
        # Start training
        self.learn = cnn_learner(data, models.resnet34, metrics=error_rate)
        defaults.device = torch.device('cuda')
        self.learn.fit_one_cycle(4)
        
        # Save model
        self.learn.save(model_name)
        self.learn.unfreeze()
        
        return self.learn
    
    
    def learning_rate(self):
        # Plot learning rate
        return self.learn.recorder.plot()
    
    def confusion_matrix(self):
        metric = ClassificationInterpretation.from_learner(self.learn)
        return metric.plot_confusion_matrix()
    
    def predict(self, img_path):
        """Predicts class of image
        
        Inputs: path to image (.png or .jpg)
        Returns: prediction for both classes
        """
        
        image = open_image(path)
        data = ItemBase(path)
        
        return self.classes, self.learn.predict(image)
    
    
class Xception():
    
    