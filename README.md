# Creditcard-fraud2

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

from pycaret.classification import *
clf1 = setup(data = df, target = 'Class')

## Introduction

Credit Card Fraud is one of the largest threats to banks and governements across the globe, causing huge financial reprocusions to businesses, we will look at the challenges faced and how to detect fraud. 
In this research document i used a data set from Kaggle, this data set consists of 284,807 transactions that happened over a 2 day period, 492 cases were fraud which shows that this data set is highly unbalanced and had only 0.172% that accounted for fraud transactions. 
The data set has 31 features, 28 labeled V1 to V28 resulting from PCA transformation resulting from confidentiality issues. 
other features are Time and Amount these represent seonds thats had lapsed between each transaction and the first transaction in this dataset. There is also a class feature which represents the label 1 for fraud transaction and o for a real transaction.




## Algorithm Selection:


