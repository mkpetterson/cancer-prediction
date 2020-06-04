import os
import sys
import shutil
import numpy
import cv2

import imageio
from scipy import misc



def dicom_to_img():
    
    # Need to be in folder with all the Mammogram dicom images
    path = '/home/maureen/Documents/Galvanize/Capstone1/Capstone3/Cancer_Prediction/data/CBIS-DDSM'
    os.chdir(path)
    dirs = [d for d in os.listdir()]

    # One dicom file in each directory, but very nested
    for d in dirs:
        path = os.path.join(os.getcwd(), d)
        for root,dirs,files in os.walk(path):
            for f in files:
                file_path = os.path.join(root,f)
                
                try:
                    dicom = dm.dcmread(file_path)
                    array = dicom.pixel_array
                    
                    # Crop 10% off all sides
                    rows, cols = array.shape
                    row_inc = int(round(0.05*rows))
                    col_inc = int(round(0.05*cols))

                    arr = array[row_inc:rows-row_inc, col_inc:cols-col_inc]  
                    
                    # Save as image. Matplotlib adds lots of crap we don't want
                    image = cv2.resize(array, (int(cols * 0.4), int(rows * 0.4)))
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                    image = np.uint8(image)
                    cv2.imwrite(f'{d}.jpg', image)
                    
                except:
                    print(d)
    return 0
    

def crop_mammograms(img_path):
    """ Crops normal mammograms, resizes, and normalizes pixels"""
    
    # Read image
    im = cv2.imread(img_path)
    image_name = os.path.splitext(img_path)[0]
    
    # Crop and normalize
    rows, cols, channels = im.shape
    row_inc = int(round(0.05*rows))
    col_inc = int(round(0.05*cols))

    arr = im[row_inc:rows-row_inc, col_inc:cols-col_inc, :] 
    image = cv2.resize(arr, (int(cols * 0.4), int(rows * 0.4)))
    cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save
    image = np.uint8(image)
    cv2.imwrite(f'{image_name}.png', image)
    
    return 0
    
    
def uniform_size(img_path):
    """ Resizes all images to have same AR and same size"""
    
    img_name = os.path.splitext(img_path)[0]
    
    # Read image
    im = cv2.imread(img_path)
    rows, cols, channels = im.shape    
    ar = rows/cols
    
    # Define best ar for MLO (need to fix cc normals)
    target_ar = 720/400
    target_width = 400
    target_height = int(round(target_width*target_ar))

    # If too many rows, crop rows
    if ar >= target_ar:

        target_rows = int(cols*target_ar)
        delta = rows - target_rows
        new_im = im[delta//2:rows-delta//2, :,:]
        rows, cols, channels = new_im.shape

    # if too many columns, crop columns
    if ar < target_ar:

        target_cols = int(rows/target_ar)    
        delta = cols - target_cols
        new_im = im[:,delta//2:cols-delta//2,:]
        rows, cols, channels = new_im.shape        

    # Resize to match minimum dimension. 
    resize = target_width/new_im.shape[1]    
    resize_im = cv2.resize(new_im, (target_width, target_height))
    
    # Renormalize to make sure all have similar brightness scale
    cv2.normalize(resize_im, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f'{img_name}_ar.png', resize_im)
            
    return 0
    

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

    
# Use all the processors!
if __name__ == '__main__':
    print('hello')
#    result = make_jpeg()
#    print(result)
#     num_processors = 8
#     pool = multiprocessing.Pool(processes=num_processors)
#     results = pool.map(make_jpeg)
#     print(results)
    

