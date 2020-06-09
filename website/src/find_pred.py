import numpy as np
import pandas as pd
import os

df = pd.read_csv('test_pred_hist_mam.csv')

def find_pred(img):
    """Output y_true, y_pred for given image"""
    
    # Load dataframe
    pred = df
        
    # Find row with image and get predictions
    row = pred[pred['image_name'] == img]
    y_true = int(row['y_true'])
    y_pred = int(row['y_pred'])
    y_proba1 = float(row['y_proba1'])
    y_proba0 = float(row['y_proba0'])
    
    return y_true, y_pred, max(y_proba1, y_proba0)
    

def compare_pred(user_pred, img):
    
    # Get true and NN predicted labels
    y_true, y_pred, y_prob_pred = find_pred(img)
    
    if user_pred == y_true:
        msg = "Correct"
        flag = 1
    else:
        msg = "Not Correct"
        flag = 0
        
    return msg        
        
    
def pick_random_image(path):
    
    files = [f for f in os.listdir(path) if '.png' in f]    
    idx = np.random.randint(0, len(files)-1)
    
    return files[idx]
