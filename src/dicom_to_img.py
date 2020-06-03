import pydicom as dm
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


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
    
    
# Use all the processors!
if __name__ == '__main__':
    print('hello')
#    result = make_jpeg()
#    print(result)
#     num_processors = 8
#     pool = multiprocessing.Pool(processes=num_processors)
#     results = pool.map(make_jpeg)
#     print(results)
    