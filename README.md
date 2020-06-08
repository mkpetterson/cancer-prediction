<img src="images/readme/cancer_A_1118_1.RIGHT_CC_sino.png" style="width: 100%; height: 20%;">
<img src="images/readme/cancer_A_1118_1.RIGHT_CC_sino.png"  width="1000" height="100">

# Cancer Detection Using Convolutional Neural Networks

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


## Data Acquisition, Exploration, and Preparation

### Data Acquisition


- The Histology data was pulled from the [Breast Cancer Histopathological Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and contains 2 different classes: >2400 benign images and >5400 malignant images at 4 different magnifications (40X, 100X, 200X, 400X).
    - Images were from a total of 82 patients, with several images at each magnification for each patient
    - Images were all 700x460 .png files with 3 channels (RGB)


- Radiographic images were pulled from two different sites: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) and the [USF Digital Mammography Database](http://www.eng.usf.edu/cvprg/Mammography/Database.html).
    - Images from The Cancer Imaging Archive were in DICOM format with accompanying metadata
    - Images from USF Database were in .LJPEG format, a lossless jpeg compression format developed by Stanford
    - Malignancies were grossly identified in some images, but not obvious in most. 


### Data Exploration

A sample of the images are shown below. Note that these thumbnails are the images after fixing the the aspect ratio and image size and not the original images. 

<center><b>Histolopathological Images</b></center>
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
          
<center><b>Radiographic Images</b></center>          
<table>
    <th>Pathology</th>
    <th>Craniocaudal (CC)</th>
    <th>Craniocaudal (CC)</th>
    <th>Mediolateral Oblique (MLO)</th>
    <th>Mediolateral Oblique (MLO)</th>
    <tr>
        <td>Normal</td>
        <td><img src="images/readme/SOB_B_F-14-14134-40-007.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_PT-14-21998AB-100-005.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_F-14-29960AB-200-013.png" width="200px"></td>
        <td><img src="images/readme/SOB_B_A-14-22549AB-400-013.png" width="200px"></td>
    </tr>
    <tr>
        <td>Cancer</td>
        <td><img src="images/readme/SOB_M_DC-14-2980-40-001.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-11031-200-001.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-13412-100-007.png" width="200px"></td>
        <td><img src="images/readme/SOB_M_DC-14-2523-400-009.png" width="200px"></td>
    </tr>
</table> 


### Data Preparation

The neural network expects all the images to have the same dimensions, which also includes color channels. The histology images required very little processing as the entire dataset was quite uniform. 

The mammograms required more extensive processing:
- Dicom images were explored using pydicom and the pixel array was extracted into a numpy array
- .LJPEG images were unpacked using a modified python script from [here](https://github.com/aaalgo/ljpeg)
- 10% of the image height and width were cropped to eliminate edge effects 
- The aspect ratio was determined and either rows or columns were minamally cropped to maintain a uniform AR across all images (no squishing)
- The images were resized to 400x720 pixels


Sinograms and Fast Fourier Transforms were applied to the images after processing. Some examples of sinogram and FFT transforms are shown below for the histology and radiographic images. 

<table>
    <th>Radiographic Images</th>
    <th>Histology Images</th>
    <tr>
        <td><img src="images/radon_fft_rad_2.png" width="500px;"></td>
        <td><img src="images/radon_fft_hist_2.png" width="500px;"></td>
    </tr>
</table>


The data was split into train/validation/test sets with the percentages of each being 70/20/10. Model performance was gauged based on both accuracy and AUC of the test set. 

## Neural Network Selection



<details>
    <summary>Raw JSON data</summary>
    <img alt="Data" src='images/json_data.png'>
</details>
    
<details>
    <summary>Raw Extracted Sample Report</summary>
    <img alt="Data" src='images/sample_report.png'>
</details>    
    
<br>    
    


## Model Performance

<b>Histology Images</b>

Using FASTAI with thre resnet34 model (34 layers), we achieved an accuracy of >87% on the test set across all the magnifications for the normal images. Looking at the AUC curves shows excellent performance across all magnifications ranging from 0.92 to 0.97. The Fourier Transforms of the data did not perform well and are barely above random guessing. 

<table>
    <th>No Transform</th>
    <th>FFT</th>
    <tr>
        <td><img src="images/hist/roc_hist_all.png" width="500px;"></td>
        <td><img src="images/hist/roc_fft_hist.png" width="500px;"></td>
    </tr>
</table>


<b>Radiographic Images</b>

The performance on the mammograms was less than the histology data. This is unsurprising given that a fair number of the positive cases are indistinguisable from the negative cases upon a cursory look at all the data. The MLO and CC datasets were trained separately and the CC cases outperformed MLO by roughly +10% on accuracy. This is likely due to the MLO images including sections of pectoral muscle, which could have increased the difficulty in training the model.  




## Conclusion

### Notes

1. https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm
2. https://www.cancer.gov/about-cancer/understanding/statistics