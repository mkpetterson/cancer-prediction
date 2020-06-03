import os
import sys
import shutil
import numpy

import imageio
from scipy import misc




def dicom_to_img():
    
    # Need to be in folder with all the Mammogram dicom images
    print(os.getcwd())

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
    
    im = imageio.imread(file)
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

    
# Use all the processors!
if __name__ == '__main__':
    print('hello')
#    result = make_jpeg()
#    print(result)
#     num_processors = 8
#     pool = multiprocessing.Pool(processes=num_processors)
#     results = pool.map(make_jpeg)
#     print(results)
    

