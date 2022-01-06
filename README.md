# Creditcard-fraud2
# A program that uses AI to analyse data


![Image Description](https://miro.medium.com/max/3000/1*3f4KQvOVZQRCsFLxvHD4rA.jpeg)


# AI - Machine-Learning-Assessment


## Software and Libraries

importing libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns

##  For t-tests and ANOVA.

import scipy.stats as stats
from matplotlib import gridspec
from imblearn.over_sampling import RandomOverSampler
%matplotlib inline
import scipy.stats as ss

## !pip install pycaret

Pycaret is open source, and it is a machine learning library for Python, this allows us to prepare our data to deploy into our model within minutes.

from pycaret.classification import *
clf1 = setup(data = df, target = 'Class')

## Introduction

Credit Card Fraud is one of the largest threats to banks and governements across the globe, causing huge financial reprocusions to businesses, we will look at the challenges faced and how to detect fraud. 
In this research document i used a data set from Kaggle, this data set consists of 284,807 transactions that happened over a 2 day period, 492 cases were fraud which shows that this data set is highly unbalanced and had only 0.172% that accounted for fraud transactions. 
The data set has 31 features, 28 labeled V1 to V28 resulting from PCA transformation resulting from confidentiality issues. 
other features are Time and Amount these represent seonds thats had lapsed between each transaction and the first transaction in this dataset. There is also a class feature which represents the label 1 for fraud transaction and o for a real transaction.




## Algorithm Selection:

Null Hypotheses



## Challenges involved in credit card fraud detection:

Large amounts of data is processed daily and when building a model to deal with this it must have the capability to keep up with it,
we deal with imbalanced dat which is the majority of the transactions 99.8% are not fraud related thus making it more complex to find the ones that are fraud related.
also we are limited to access to data as it is mainly private, and also not all fraud cases are reported so this destorts data results.
Our model must detect fraud as fast as possible, we can also deal with imbalance if required. The model must be easy to read and simple. 


## Loading Data

Loading the data we used this website https://www.kaggle.com/mlg-ulb/creditcardfraud/data#, we then downloaded the cardfraud.csv file and began work on it. we beagn some basic analysis of the dat to get a better understanding of what the data was. Some of the data was easy to read such as Time, Amount etc.. but due to confidentiality we had V1 - V28 which was a result of an application of PCA transformation applied to the original ones. 

Time = the number of seconds that have lapsed between  this transaction to the  first one.

Amount = is the transaction amount

0 = others

1 = fraud cases

We begin by looking at both the top 5 lines of data and the bottom 5 lines.Then we take a look of the shape of the data. 
From the above data we can see that from looking at the data in Time, there has been 284807 transactions over a 2 day period

Data Processing & Understanding

The dataset is imbalanced towards a feature, as various banks have adopted dufferent mechanisms making it harder for them to have a data breach. But we will still see some vulnerability in the system, and thus this explains why most transactions are normal and a smaller percent are fraudulent.




## References

https://github.com/amancodeblast/Credit-Card-Fraud-Detection/blob/master/Credit%20card%20fraud%20detection%20using%20Pycaret_implementation.ipynb
https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3
https://medium.com/analytics-vidhya/credit-card-fraud-detection-in-python-using-scikit-learn-f9046a030f50
https://github.com/Maria1709/machine-learnig-assessment/blob/master/Boston1.0%20(7)%20(1).ipynb

file:///C:/Users/maria/AppData/Roaming/jupyter/runtime/nbserver-6568-open.html
https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models

https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/

http://localhost:8888/?token=e37e222495797b8c0ee4f60e92af5b68117606aecff2eee2
https://github.com/amancodeblast/Credit-Card-Fraud-Detection/blob/master/Credit%20card%20fraud%20detection%20using%20Pycaret_implementation.ipynb
https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3
https://medium.com/analytics-vidhya/credit-card-fraud-detection-in-python-using-scikit-learn-f9046a030f50
https://github.com/Maria1709/machine-learnig-assessment/blob/master/Boston1.0%20(7)%20(1).ipynb
/
file:///C:/Users/maria/AppData/Roaming/jupyter/runtime/nbserver-6568-open.html
https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models

https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/

http://localhost:8888/?token=e37e222495797b8c0ee4f60e92af5b68117606aecff2eee2
    
https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/








