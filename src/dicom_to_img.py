import pydicom as dm
import numpy as np
import os
import sys
import multiprocessing
import matplotlib.pyplot as plt


def make_jpeg():
    
    # Navigate to folder with Mammograms and get all directories
    os.chdir('CBIS-DDSM')
    dirs = [d for d in os.listdir()]

    # One dicom file in each directory, but very nested
    for d in dirs:
        path = os.path.join(os.getcwd(), d)
        for root,dirs,files in os.walk(path):
            for f in files:
                file_path = os.path.join(root,f)
             #  flip_l_to_r(file_path)
                try:
                    dicom = dm.dcmread(file_path)
                    array = dicom.pixel_array
                    plt.imshow(array)
                    plt.savefig(f'{d}.png', dpi=350)
                except:
                    print(d)
    return 0
                    

# Use all the processors!
if __name__ == '__main__':
    result = make_jpeg()
    print(result)
#     num_processors = 8
#     pool = multiprocessing.Pool(processes=num_processors)
#     results = pool.map(make_jpeg)
#     print(results)
    