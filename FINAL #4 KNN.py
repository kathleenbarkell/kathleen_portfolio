#!/usr/bin/env python
# coding: utf-8

# # Final Supervised ML: KNN
# ## Kathleen Harris

# In[15]:


# Import Python Libraries: NumPy and Pandas

import pandas as pd
import numpy as np

# Import Libraries & modules for data visualization

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

# Import scikit-Learn module to split the dataset into train/ test sub-datasets

from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL 
# evaluation & validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Import scikit-Learn module classification report to later use for information 
# about how the system try to classify/label each record

from sklearn.metrics import classification_report


# ## 1) Load the Data

# In[2]:


# Specify location of the dataset

filename = './data/pima_diabetes.csv'

# Load the data into a Pandas DataFrame

df = pd.read_csv(filename)


# In[3]:


# Preview the dataframe to verify it loaded
df.head()


# ## 2) Preprocess the Dataset

# In[6]:


# Mark zero values as missing or NaN
df[[ 'preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age']] = df[['preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age' ]].replace(0,np.NaN)


# In[7]:


# count the number of NaN values in each column
print (df.isnull().sum())


# In[8]:


# replace missing values with the mean
df=df.fillna(df.mean())


# In[9]:


# verify there is no missing data
print (df.isnull().sum())


# ## 3) EDA

# In[10]:


# get the dimensions of the dataset

print("Shape of the dataset(rows, columns):",df.shape)


# In[11]:


#get the data types of all the variables / attributes in the data set

print(df.dtypes)


# In[12]:


#return the summary statistics of the numeric variables in the data set

print(df.describe())


# In[13]:


#class distribution i.e. how many records are in each class
# looking at the csv file, class looks like a dummy variable
# 0 - no diabetes, 1 - yes diabetes

print(df.groupby('class').size())


# ### Histogram - first visualization

# In[16]:


# Plot histogram for each variable.

df.hist(edgecolor= 'black',figsize=(14,12))
plt.show()


# #### <div class="alert alert-info">Plas and pres are fairly normally distrubuted.  The remaining (except class) are right skewed</div>

# ### Boxplot - second visualization

# In[17]:


df.plot(kind="box", subplots=True, layout=(5,3), sharex=False, figsize=(20,18))
plt.show()


# ### Scatterplot - third generalization

# In[18]:


# generate scatter plot matrix of each numeric variable in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# #### <div class="alert alert-info"> I did a scatter plot this time to compare to the pair plot w/ color I used in problem 2.  I prefer the pair plot with color as it has the added benefit of also seeing how each data point is classified (diatbetic or not).</div>

# ## 4) Separate the Dataset into Input & Output NumPy Arrays

# In[19]:


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
# the class variable is indexed at 8, so I revised the code

Y = array[:,8]


# ## 5) Spilt into Input/Output Array into Training/Testing Datasets

# In[20]:


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


# To remove the future warning, I will:
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#build the model
model = KNeighborsClassifier()
# train the model using the training sub-dataset
model.fit(X_train, Y_train)
#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print("Classification Report: ", "\n", "\n", report)


# ### Remember, F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar costs. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

# #### <div class="alert alert-info">In this case, we have an uneven class distribution, so F1 will be the better score to consider.</div>

# ## 7) Score the Accuracy of the Model

# In[23]:


#score the accuracy leve
result = model.score(X_test, Y_test)
#print out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# #### <div class="alert alert-info"> The accuracy of this model is less than the logisitc regression model which was 75.984%.  However, since the values of false positivies and false negatives are not the same (symmetric), we need to examine other parameters such as F1 to evaluate the model performance.</div>

# ## 8/9) Prediction

# In[24]:


# First Record using averages found during EDA rounding to nearest hundredth:

# preg = 4.49, plas=121.69, pres=72.41, skin=29.15, test=155.55, mass=32.46
# pedi=0.47, age = 33.24

model.predict([[4.49, 121.69, 72.41, 29.15, 155.55, 32.46, 0.47, 33.24]])


# #### <div class="alert alert-info">The model classifies a patient with the submitted parameters (based on the averages of the variables), as having diabetes.  Note that this is different than the logistic outcome.</div>

# In[25]:


# Second Prediction:
# It is generally known that weight affects a diabetes diagnosis, so I will use  
# the max value for mass keeping the mean values for the remaining variables, 
# and see if the model yeilds a positive diabetes diagnosis.

model.predict([[4.49, 121.69, 72.41, 29.15, 155.55, 67.1, 0.47, 33.24]])


# #### <div class="alert alert-info">The model classifies a patient with the submitted parameters as having diabetes.  Note that this is the same as the logistic outcome.</div>

# ## 10) Evaluate the model using the 10-fold cross-validation technique.

# In[26]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10

# use the same random seed value
seed = 7

# split the whole dataset into folds
kfold = KFold(n_splits, random_state=seed, shuffle=True)

# for logistic regression, we can use the accuracy level to evaluate the model/algorithm
scoring = 'accuracy'

# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# print the evaluation results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# #### <div class="alert alert-info">Note that the evaluation of the KNN model is less than the logistic model which was 77.2%</div>

# In[ ]:




