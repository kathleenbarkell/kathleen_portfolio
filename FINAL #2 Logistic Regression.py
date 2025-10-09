#!/usr/bin/env python
# coding: utf-8

# # Final Supervised ML: Logistic Regression
# ## Kathleen Harris

# In[1]:


# Import Python Libraries: NumPy and Pandas

import pandas as pd
import numpy as np


# In[2]:


# Import Libraries & modules for data visualization

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Import scikit-Learn module for the algorithm/modeL: Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[4]:


# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split


# In[5]:


# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL 
# evaluation & validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[6]:


# Import scikit-Learn module classification report to later use for information 
# about how the system trys to classify/lable each record

from sklearn.metrics import classification_report


# ## 1) Load Data

# In[7]:


# Specify location of the dataset

filename = './data/pima_diabetes.csv'

# Load the data into a Pandas DataFrame

df = pd.read_csv(filename)


# In[8]:


# Preview the dataframe to verify it loaded
df.head()


# ## 2) Preprocess the Dataset

# In[9]:


# Mark zero values as missing or NaN
df[[ 'preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age']] = df[['preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age' ]].
replace(0,np.NaN)


# In[10]:


# count the number of NaN values in each column
print (df.isnull().sum())


# In[11]:


# replace missing values with the mean
df=df.fillna(df.mean())


# In[12]:


# verify there is no missing data
print (df.isnull().sum())


# ## 3) EDA

# In[13]:


# get the dimensions of the dataset

print("Shape of the dataset(rows, columns):",df.shape)


# In[14]:


#get the data types of all the variables / attributes in the data set

print(df.dtypes)


# In[15]:


#return the summary statistics of the numeric variables in the data set

print(df.describe())


# In[16]:


#class distribution i.e. how many records are in each class
# looking at the csv file, class looks like a dummy variable
# 0 - no diabetes, 1 - yes diabetes

print(df.groupby('class').size())


# ### Histogram - first visualization

# In[17]:


# Plot histogram for each variable.

df.hist(edgecolor= 'black',figsize=(14,12))
plt.show()


# #### plas and pres are fairly normally distrubuted
# #### the remaining (except class) are right skewed

# ### Boxplot - second visualization

# In[18]:


df.plot(kind="box", subplots=True, layout=(5,3), sharex=False, figsize=(20,18))
plt.show()


# ### Pair Plot with color - third visualization

# In[19]:


sns.pairplot(df, hue='class', height=3, aspect= 1);


# ## 4) Separate the Dataset into Input & Output NumPy Arrays

# In[20]:


# store dataframe values into a numpy array

array = df.values

# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 5 
# these are the independent variables or predictors
# we don't have an id column, so we want all 8 independent variables
# revised code below with columns from 0 to 8

X = array[:,0:8]

# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
# the class variable is indexed at 8, so revise the code

Y = array[:,8]


# ## 5) Spilt into Input/Output Array into Training/Testing Datasets

# In[21]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%

test_size = 0.33

#selection of records to include in each data sub-dataset must be done randomly
# seed is the randomness value

seed = 7

#split the dataset (input and output) into training / test datasets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ## 6) Build and Train the Model

# In[22]:


#build the model

model = LogisticRegression(random_state=seed, max_iter=1000)

# train the model using the training sub-dataset

model.fit(X_train, Y_train)

#print the classification report

predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print("Classification Report: ", "\n", "\n",report)


# In[25]:


#remember from Data analtyics 2:
# Precision- Accuracy of positive predictions.
# Precision = TP/(TP + FP)

# Recall- Fraction of positives that were correctly identified.
# Recall = TP/(TP+FN)

# F1 - the closer to 1 the better
# F1 scores are lower than accuracy measures as they embed precision and 
# recall into their computation. As a rule of thumb, the weighted average of F1 
# should be used to compare classifier models, not global accuracy.
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)

# Support -  number of actual occurrences of the class in the specified dataset.


# #### https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397

# ## 7) Score the Accuracy of the Model

# In[26]:


#score the accuracy level

result = model.score(X_test, Y_test)

#print out the results

print(("Accuracy: %.3f%%") % (result*100.0))


# ## 8/9) Prediction

# In[27]:


# First Record using averages found during EDA rounding to nearest hundredth:

# preg = 4.49, plas=121.69, pres=72.41, skin=29.15, test=155.55, mass=32.46
# pedi=0.47, age = 33.24

model.predict([[4.49, 121.69, 72.41, 29.15, 155.55, 32.46, 0.47, 33.24]])


# #### <div class="alert alert-info">The model classifies a patient with the submitted parameters (based on the averages of the variables), as NOT having diabetes.</div>

# In[28]:


# Second Prediction:
# It is generally known that weight affects a diabetes diagnosis, so I will use  
# the max value for mass keeping the mean values for the remaining variables, 
# and see if the model yeilds a positive diabetes diagnosis.

model.predict([[4.49, 121.69, 72.41, 29.15, 155.55, 67.1, 0.47, 33.24]])


# #### <div class="alert alert-info">The model classifies a patient with the above submitted parameters as having diabetes.</div>

# ## 10) Evaluate the Model using the 10-fold Cross-Validation Technique.

# In[29]:


# Evaluate the algorithm and specify the number of times of repeated splitting

n_splits=10

#Verify the random seed is the same as previously used so that the same subsets 
# can be obtained for each time the process is repeated

seed=7

kfold=KFold(n_splits, random_state=seed, shuffle=True)

# for logistic regression, we can use the accuracy level to evaluate the model 

scoring="accuracy"

#train the model and run K-fold cross validation to validate / evaluate the model

results=cross_val_score (model, X, Y, cv=kfold, scoring=scoring)

# print the evaluation results.  The result is the average of all the results 
# obtained from the K-fold cross validation

print("Accuracy: %.3f (%.3f)"% (results.mean(), results.std()))


# #### <div class="alert alert-info">The cross-validation shows the model is 77.2% accurate with a standard deviation of 0.034 </div>

# In[ ]:




