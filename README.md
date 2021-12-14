# Creditcard-fraud2

![Image Description](https://miro.medium.com/max/3000/1*3f4KQvOVZQRCsFLxvHD4rA.jpeg)


# AI - Machine-Learning-Assessment


## Software and Libraries

#importing libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
# For t-tests and ANOVA.
import scipy.stats as stats
from matplotlib import gridspec
from imblearn.over_sampling import RandomOverSampler
%matplotlib inline
import scipy.stats as ss
!pip install pycaret
from pycaret.classification import *
clf1 = setup(data = df, target = 'Class')
