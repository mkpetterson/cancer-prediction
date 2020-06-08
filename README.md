# Cancer Detection using Convolutional Neural Networks

![badge](https://img.shields.io/badge/last%20modified-june%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)


## Table of Contents

- <a href="https://github.com/mkpetterson/Cancer_Prediction#Introduction">Introduction</a> 
- <a href="https://github.com/mkpetterson/UFO_sightings#data-acquition-and-exploration">Data Acquisition and Exploration</a> 
- <a href="https://github.com/mkpetterson/UFO_sightings#neural-network-selection">Neural Network Selection</a> 
- <a href="https://github.com/mkpetterson/UFO_sightings#model-performance">Model Performance</a> 
- <a href="https://github.com/mkpetterson/UFO_sightings#conclusion">Conclusion</a>


## Introduction

Out of box performance:

FastAI: on 100X histology dataset: 92% on validation set
Transfer Learning Xception: 88% on training and 75% on validation (overfitting?) Run over 3 epochs
Simple CNN: 69% on validation and training. Ended up abandoning this model.


UFO sightings occur with relative frequency all across the United States. The sighted UFOs have various shapes and the sightings last for varying amounts of time. Using the UFO sighting database, we evaluated several characteristics of the sightings and used Natural Language Processing (NLP) to analyze the descriptions and see what commonalities all the descriptions had. 

The data was pulled from the [The National UFO Reporting Center Online Database](http://www.nuforc.org/webreports.html).  


## Data Acquisition and Exploration

### Data Acquisition

The raw data was 2.5GB and required a decent amount of preparation prior to analysis. We downloaded a zipped json file that included the raw HTML for each individual sighting.

Cleaning and preparation methods included:

- Extracting the unique observation ID, date, time, location, shape and text description of the sightings
    - First we used Beautiful Soup's html parser to extract data contained within specific HTML tags
    - Limited data to about 15,000 in order for it to not run forever
    - Regular expressions were utilized to extract the exact terms we needed to run analyis on the different features
- Separating the text description from the follow-up notes
- Putting the information into a pandas datafram for easier analysis

<details>
    <summary>Raw JSON data</summary>
    <img alt="Data" src='images/json_data.png'>
</details>
    
<details>
    <summary>Raw Extracted Sample Report</summary>
    <img alt="Data" src='images/sample_report.png'>
</details>    
    
<br>    
    
The cleaned up pandas dataframe is shown below
    
  <img src='images/initial_df.png'>


### Exploratory Data Analysis

The sightings described the UFOs as various different shapes, including circles, chevrons, lights, or fireballs. The duration of the sightings lasted from a few seconds to many minutes. 


**Shapes and Duration**
<img alt="shapes" src='images/shape_duration.png' style='width: 600px;'>


The time of day for the observations were also interesting. Sightings tended to be higher in the early morning or evening hours, which makes sense as UFO lights will not be as visible during daylight hours. It's also possible many people mistake planets, satellites, or planes as UFOs.  

<img alt="timeofday" src='images/time_of_day.png'>

**State**

We got a count of the states and sightings. It seems California is number one for UFO sightings.

<img alt="state_count" src='images/state_counts.png'>


## Natural Language Processing
The data was analyzed using a combination of nltk packages and sklearns CountVectorizer/TFIDFVectorizer to analysis the most common words within the observations. We also used topic modeling to extract latent features of the text. The pipeline used on each observation was:

1. Tokenization of text observations 
2. Stop Words removal (standard English)
3. Lemmitization using nltk WordNetLemmatizer
4. TFIDFVectorizer to get the relative word strength
5. Topic Modeling using Non-negative Matrix Factorization (NMF)

Fitting the Model:
<img alt="vanilla topics" src='images/vanilla_model.png'>

Top 10 Topics:
<img alt="vanilla topics" src='images/vanilla_topics.png'>





Using this pipeline allowed us to visualize the most common words for the observations. 

<b>UFO Sightings</b>
<img alt="ufowords" src='images/UFO_words.png'>

<b>Notes on the UFO Sightings</b>
<img alt="ufonoteswords" src='images/UFO_notes_words.png'>

<b>Bigfoot Sightings</b>
<img alt="bigfootwords" src='images/bigfoot_words.png'>

## Summary and Key Findings



