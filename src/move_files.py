import os
import sys
import re
import subprocess as sub

import imageio

import PIL
from PIL import Image

current_dir = os.getcwd()
path = os.path.join(current_dir, 'normal_03/')
print(path)
    
for root,dirs,files in os.walk(path):
    for d in dirs:
        new_path = os.path.join(path,d)
        os.chdir(new_path)
        files = [f for f in os.listdir() if '.LJPEG' in f]
               
        # Loop through files
        for f in files:
            file_path = os.path.join(new_path,f)
            file_name = os.path.splitext(f)[0]
            print(file_path)
            sub.call(f'{current_dir}/ljpeg.py {file_path} {file_name}.jpg --visual --scale 1.0')


    
def resize_image_500pix(file):
    
    file_name = os.path.splitext(file)[0]
    
    basewidth = 500
    img = Image.open(file)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(f'{file_name}r1.jpg')
    
    return 0
    
       
