import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
import imageio
import scipy
from skimage.transform import radon, rescale



def fft(img_path):
    """ FFT of histology images (.png)"""
    
    img_name = os.path.splitext(img_path)[0]
    
    
    # Open image and sum along all color channels
    image = imageio.imread(img_path).sum(axis=2) # for 3 channel images
#    image = rescale(image, scale=0.8, mode='reflect', multichannel=True)
    
    # FFT
    f = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f)
    fft = np.log(np.abs(f_shift))
    
    # Save image
    imageio.imsave(f'{img_name}_fft.png', fft)
    
    return None


def sinogram(img_path):    
    
    img_name = os.path.splitext(img_path)[0]  
    
    # Open image and rescale (sinogram is slow)
    image = imageio.imread(img_path).sum(axis=2) # for 3 channel images
    
    # Rescale if dealing with large image, otherwise ignore
 #   image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    
    # Make sinogram
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sino = radon(image, theta=theta, circle=True)
    
    # Save image
    imageio.imsave(f'{img_name}_sino.png', sino)
    
    return None