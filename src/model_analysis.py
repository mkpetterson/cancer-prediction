import numpy as np
import pandas as pd



def predictions(pred_tensor):
    """ Turn FastAI tensor into y_pred and y_proba"""
    
    
    arr = np.asarray(pred_tensor)
    
    y_pred = arr[:,1].astype(int)
    proba_tensor = [np.array(i) for i in arr[:,2]]
    y_proba = np.asarray(proba_tensor)
    
    return y_pred, y_proba


def make_dataframe(name, filenames, y_true, y_pred, y_proba):
    
    
    results = pd.DataFrame({'Filenames': pd.Series(filenames), 
                           'y_true': pd.Series(y_true),
                          'y_pred': pd.Series(y_pred),
                          'y_proba0': y_proba[:,0],
                          'y_proba1': y_proba[:,1]})
    
    results.to_csv(f'{name}.csv')
    
    return None


