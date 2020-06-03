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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model


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
    def __init__(self, img_width, img_height, channels=3, classes=2):
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.classes = classes
        
        # Set up datagen
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                      horizontal_flip=True) 
        self.val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    
    def create_transfer_model(self, input_size, n_categories, weights = 'imagenet'):
        # note that the "top" is not included in the weights below
        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        
        self.model = base_model.output
        self.model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def change_trainable_layers(self, model, trainable_index):
        for layer in self.model.layers[:trainable_index]:
            layer.trainable = False
        for layer in self.model.layers[trainable_index:]:
            layer.trainable = True
            
        return None
        
        
    def compile_model(self, train_path, val_path):
        """Compile the model by training only the head of Xception model"""
        
        # Get generators and parameters for training/validation data
        train_generator = self.train_datagen.flow_from_directory(train_path, 
                                                       target_size=(self.img_width,self.img_height), 
                                                       batch_size=16)
        validation_generator = self.val_datagen.flow_from_directory(val_path, 
                                                              target_size=(self.img_width,self.img_height), 
                                                              batch_size=16)

        self.n_train = len([f for f in os.listdir(train_path)])
        self.n_val = len([f for f in os.listdir(val_path)])        
        
        
        # Create model, make head trainable, and compile
        model = create_transfer_model((self.img_width,self.img_height,self.channels),self.classes) 
        _ = change_trainable_layers(self.model, 132)
        self.model.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

        return None



    def fit(self):
        """Fit model"""
        
        self.model.fit_generator(train_generator, steps_per_epoch=self.n_train//16, epochs=3, 
                                 validation_data=self.val_generator, validation_steps=self.n_val//16)        
        
        model.save_weights('models/weights.h5')
#        model.save('models/transfermodel.h5')
        
        return None
    
    
    def predict(self, test_path):
        """Predict on directory of images"""
        
        test_generator = self.test_datagen.flow_from_directory(test_path,
                                                 target_size=(self.img_width,self.img_height),
                                                 batch_size=16)
        predictions = model.predict(test_generator)

        return predictions