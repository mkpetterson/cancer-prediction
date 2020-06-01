import os
import sys
import shutil
import numpy

import imageio
from scipy import misc



def sort_by_mag(root, file):
    """ Sorts files by magnification"""
    
    print(os.getcwd())
    file_old_path = os.path.join(root, file)
    
    if '40X' in root:
        mag_path = '40X'
    elif '100X' in root:
        mag_path = '100X'
    elif '200X' in root:
        mag_path = '200X'
    else:
        mag_path = '400X'
        
    file_new_path = os.path.join(mag_path, file)
    print(file_old_path, file_new_path)
    shutil.move(file_old_path, file_new_path)
    
    return 0

def sort_images(path):
    """Sorts histology files. Need to be in either benign or malignant folder"""
    
    for root,dirs,files in os.walk(path):
        for f in files:
            if '.png' in f:
                sort_by_mag(root, f)
            else:
                print(f)
    return 0



def crop_image(file):
    """ Crops malignant images"""
    
    file_name = os.path.splitext(file)[0]
    
    im = imageio.imread(file)#, as_gray=True)#, pilmode="L")
    im_cropped = im[200:1200, 800:1400, :]
#    print(im_cropped.shape)
    
    fig = plt.figure(frameon=False, figsize=(3,5))
    
    # Make content fill full figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(im_cropped, aspect='auto', cmap='gray')
    fig.savefig(f'{file_name}.png', bbox_inches='tight', dpi=350)
    plt.close(fig)

    return 0