# Cancer Detection using Convolutional Neural Networks

![badge](https://img.shields.io/badge/last%20modified-june%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)


## Table of Contents

- <a href="https://github.com/mkpetterson/Cancer_Prediction#Introduction">Introduction</a> 
- <a href="https://github.com/mkpetterson/Cancer_Prediction#data-acquition-and-exploration">Data Acquisition and Exploration</a> 
- <a href="https://github.com/mkpetterson/Cancer_Prediction#neural-network-selection">Neural Network Selection</a> 
- <a href="https://github.com/mkpetterson/Cancer_Prediction#model-performance">Model Performance</a> 
- <a href="https://github.com/mkpetterson/Cancer_Prediction#conclusion">Conclusion</a>


## Introduction

Cancer is the second leading cause of death in the United States (1), with over 1.7 million people expected to be diagnosed this year (2). Among the different types of cancer, lung and breast are the most common. Radiography is one of the first diagnostic tests used to diagnose tumors and subsequent biopsy can determine if a tumor is malignant of benign. Early detection is key to improving outcomes. 

The advent of image classsification thorugh machine learning has given the medical industry another tool with which to help diagnose patients. While advanced imaging algorithms and neural networks cannot replace medical professionals, this technology can help guide the diagnosis. 

The goal of this project was to build a simple breast cancer image classifier using convolutional neural networks using both histology (microscopic) images from biopsies and radiographic images (mammograms). Additionally, Radon transforms and Fast Fourier Transforms were applied to see if they could augment the predictions. 2 different CNNs were investigated: Tensorflow and Pytorch-based FastAI. 



Out of box performance:

FastAI: on 100X histology dataset: 92% on validation set
Transfer Learning Xception: 88% on training and 75% on validation (overfitting?) Run over 3 epochs
Simple CNN: 69% on validation and training. Ended up abandoning this model.

The data was pulled from both [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) and the USF 


## Data Acquisition and Exploration

### Data Acquisition


- The Histology data was pulled from the [Breast Cancer Histopathological Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and contains 2 different classes: >2400 benign images and >5400 malignant images at 4 different magnifications (40X, 100X, 200X, 400X).

- Radiographic images were pulled from two different sites: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) and the [USF Digital Mammography Database](http://www.eng.usf.edu/cvprg/Mammography/Database.html).

<table>
    <th>Tumor Type</th>
    <th>40X</th>
    <th>100X</th>
    <th>200X</th>
    <th>400X</th>
    <tr>
        <td>Benign</td>
        <td><img src="images/readme/SOB_B_F-14-14134-40-007.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_PT-14-21998AB-100-005.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_F-14-29960AB-200-013.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_A-14-22549AB-400-013.png" width="200px"></td>
    </tr>
    <tr>
        <td>Malignant</td>
        <td><img src="images/readme/SOB_M_DC-14-2980-40-001.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-11031-200-001.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-13412-100-007.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-2523-400-009.png" width="200px"></td>
    </tr>
</table>    
            


<details>
    <summary>Raw JSON data</summary>
    <img alt="Data" src='images/json_data.png'>
</details>
    
<details>
    <summary>Raw Extracted Sample Report</summary>
    <img alt="Data" src='images/sample_report.png'>
</details>    
    
<br>    
    


### Data Exploration


## Neural Network Selection


## Model Performance



## Conclusion

### Notes

1. https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm
2. https://www.cancer.gov/about-cancer/understanding/statistics