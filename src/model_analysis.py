import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve



def predictions(pred_tensor):
    """ Turn FastAI tensor into y_pred and y_proba"""   
    
    # Turn into numpy array
    arr = np.asarray(pred_tensor)
    
    # Pull out prediction label and class probabilities
    y_pred = arr[:,1].astype(int)
    proba_tensor = [np.array(i) for i in arr[:,2]]
    y_proba = np.asarray(proba_tensor)
    
    return y_pred, y_proba


def make_dataframe(name, filenames, y_true, y_pred, y_proba):
    """ Makes dataframe from data and saves as csv """  
    
    results = pd.DataFrame({'Filenames': pd.Series(filenames), 
                           'y_true': pd.Series(y_true),
                          'y_pred': pd.Series(y_pred),
                          'y_proba0': y_proba[:,0],
                          'y_proba1': y_proba[:,1]})
    
    results.to_csv(f'{name}.csv')
    
    return None


def make_confusion(name, y_true, y_pred, y_proba):
    """ Plot and save confusion matrix """
    
    # Make confusion plot
    fig, ax = plt.subplots()
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns_plot = sns.heatmap(cf_matrix, annot=True, fmt='5.0f', cmap='Blues')
    
    # Get sns plot onto matplotlib figure for saving
    fig = sns_plot.get_figure()
    plt.title(f'Confusion Matrix: {name}')
    fig.savefig(f'cf_{name}.png', dpi=350) 
    plt.close()   
    
    return None

def plot_roc(ax, label, y_true, y_proba):
    """ Plots ROC curve given true labels and predicted probabilities"""
    
    # Get AUC score and fpr, tpr, thresholds
    auc = roc_auc_score(y_true, y_proba[:,1])
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:,1])    
    
    # Plot on ax
    ax.plot(fpr, tpr, label=f'{label} AUC: {auc:2.2f}')
    ax.legend(loc='lower right')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")   

    return ax

