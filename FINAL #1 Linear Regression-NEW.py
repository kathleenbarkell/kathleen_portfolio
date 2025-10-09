#!/usr/bin/env python
# coding: utf-8

# # Final Supervised ML:  Linear Regression
# # Kathleen Harris

# In[2]:


import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix


from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



import seaborn as sns
sns.set(color_codes=True)

import matplotlib.pyplot as plt


# ## 1) Load the Data

# In[4]:


# loading the file
housingfile = './data/housing_boston.csv'


# In[5]:


# loading the data into a Pandas dataframe
df= pd.read_csv (housingfile, header=None)


# In[6]:


# verifying the data was loaded by checking the first 5 rows
df.head()


# ### Label the Columns

# In[7]:


col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX','RM', 'AGE', 'DIS', 'RAD', 
             'TAX', 'PTRATIO', 'AA', 'LSTAT', 'MEDV']


# In[8]:


df.columns = col_names


# In[9]:


#  Look at the first 5 rows of data to verify the column names

df.head()


# ## 2) Preprocess the Dataset
# ### Clean the data:  Find and mark missing values

# In[8]:


df.isnull().sum()

# There are no missing data points


# ## 3) EDA

# In[10]:


# determine the number of records and variables
# preview the data types of all variables
print(df.shape)
print(df.dtypes)


# In[10]:


# View the summary statistics
print(df.describe())


# ### Create a Histogram - first visualization

# In[11]:


df.hist(edgecolor= 'black',figsize=(14,12))
plt.show()


# ### Creating a Box Plot - second visualization

# In[12]:


df.plot(kind="box", subplots=True, layout=(5,3), sharex=False, figsize=(20,18))
plt.show()


# ### Correlation Analysis and Feature Selection

# In[13]:


# viewing 3 decimals worth of the correlation

pd.options.display.float_format = '{:,.3f}'.format
df.corr()


# ### Heatmap - third visualzation

# In[14]:


# Creating a heat map to better visualize the correlations

plt.figure(figsize =(16,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[11]:


# After reviewing the correlation of the variables from the heatmap, I want to
# reduce the number of variables by making a subset making sure to include
# my target variable, MEDV.

# CRIM & TAX, CRIM & RAD, TAX & RAD, MEDV & RM all have strong positive 
# correlations to each other

df2= df[['CRIM','RAD', 'TAX', 'RM', 'MEDV']]


# #### Recall the descriptions of the variables:
# #### MEDV:  This is the median avlue of owner-occupied home in thousands
# #### CRIM:  This is the per capita crime rate by town
# #### RAD:  This is the index of accessibility to radial highways
# #### TAX:  This the full-value property-tax rate per $10,000
# #### RM: This is the average number of rooms per dwelling

# In[12]:


# Reviewing correlations of my selected variables

df2.corr()


# In[13]:


# CRIM & RAD have a super strong positive realtionship at 0.9 
# which may cause some overfitting.

# I'm concerend with variables that have a strong relationship to each other
# causing multicollinearity issues.
# I'm going to examine variables that have a strong relationships with 
# MEDV specifically.
# I also orignally thought we were to have 5 total variables, but after
# class I adjusted this to be 5 independent variables and 1 target variable

df3= df[['INDUS','LSTAT','RM', 'PTRATIO', 'TAX', 'MEDV']]


# #### Recall the descriptions of the variables:
# #### INDUS:  porportion of residential land zone for lots larger than 25,000 sq ft
# #### LSTAT:  percentage lower status of the population
# #### PTRATIO:  pupil-teacher ratio by town

# In[14]:


# Reviewing correlations of the newest selected variables.
df3.corr()


# ## 4) Separate the Dataset into Input & Output NumPy Arrays

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# Store the dataframe values into a numPy array

array= df3.values

# Separate the array into input and output components by slicing
# Note we have additional variables now, so we need to include 
# an extra column!
# For X (input) [:,5] --> All the rows and columns from 0 up to 5

X = array [:, 0:5]

# For Y (output) [:5] --> All the rows in the last column (MEDV)

Y = array [:,5]


# ## 5) Spilt into Input/Output Array into Training/Testing Data

# In[17]:


# Split the dataset --> training sub-dataset:  67%, and test sub-dataset:  33%

test_size = 0.33

# Selection of records to include in which sub-dataset must be done randomly - 
# use the for seed radomization

seed = 7

# Split the dataset (both input & output) into training/testing datasets
# if random_state = None : Calling the function multiple times will produce 
# different results.
# if random_state = Integer : Will produce the same results across 
# different calls

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, 
                                                   random_state=seed)


# ## 6) Build and Train the Model

# In[20]:


# Build the model

model=LinearRegression()

# Train the model using the training sub-dataset

model.fit(X_train, Y_train)

#Print out the coefficients and the intercept
# Print intercept and coefficients
# are the variables statistically significant
# interdept = mean (average) value of Y
# if the value is less than 0.05: there is a strong relationship between the 
# variable and the target  

print ("Intercept:", model.intercept_)
print ("Coefficients:", model.coef_)


# In[19]:


# If we want to print out the list of the coefficients with their 
# correspondent variable name
# Pair the feature names with the coefficients

names_3 = ['INDUS','LSTAT','RM', 'PTRATIO', 'TAX', 'MEDV']

coeffs_zip = zip(names_3, model.coef_)

# Convert iterator into set

coeffs = set(coeffs_zip)

# Print (coeffs)

for coef in coeffs:
    print (coef, "\n")


# In[21]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# ## 7) Calculate R-squared

# In[22]:


R_squared = model.score(X_test, Y_test)
print("R-squared: ", R_squared)


# #### <div class="alert alert-info">This r squared value is much better than the homework; it's also slightly better than the model I previously had with only 4 independent variables which had an r-squared value of 0.619 using some different variables (LSTAT, RM, INDUS, and AGE).  However, it's still not great.  I'm curious to see the validation rating.</div>

# ## 8) Prediction

# ### Predict the "Median value of owner-occupied homes in 1000 dollars"
# #### First Record using averages found during EDA:
# #### INDUS = 10.3
# #### LSTAT = 11.44
# #### RM = 6 (rounded to nearest whole number)
# #### PTRATIO = 18.24
# #### TAX = 377.44

# In[26]:


model.predict([[10.3,11.44,6,18.24,377.44]])


# #### <div class="alert alert-info">Given the above parameters, the model predicts the median value of owner-occuped homes in the given suburb to be approximately $21,529.</div>

# #### Second Record:
# #### INDUS = 10
# #### LSTAT = 11
# #### RM = 4 (closer to min value)
# #### PTRATIO = 18.24
# #### TAX = 377.44

# In[27]:


model.predict([[10,11,4,18.24,377.44]])


# #### <div class="alert alert-info">Given the above parameters, changing only the number of rooms (RM), the model predicts the median value of owner-occuped homes in the given suburb to be approximately $9,171.</div>

# ## 9) Evaluate the model using 10-fold cross validation

# In[24]:


# Evaluate the algorithm
# Specify the K-size

num_folds = 10

# Fix the random seed
# must use the same seed value so that the same subsets can be obtained for 
# each time the process is repeated

seed = 7

# Split the whole data set into folds

kfold= KFold(n_splits=num_folds, random_state=seed, shuffle=True)

# For Linear regression, we can use MSE (mean squared error) value to evaluate the 
# model/algorithm

scoring = 'neg_mean_squared_error'

# Train the model and run K-foLd cross-validation to validate/evaluate the model

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Print out the evaluation results
# Result: the average of all the results obtained from the k-fold 
# cross validation

print("Average of all results from the K-fold Cross Validation, using negative mean squared error:",results.mean())


# #### <div class="alert alert-info">This MSE (approx -25) is better than the MSE value from the homework (-64) and from my previous model (-33); thus, this model is an improvement and stronger than previous models.</div>

# In[25]:


# Now I'm going to look at the explained variance

num_folds = 10

seed = 7

# Split the whole data set into folds

kfold= KFold(n_splits=num_folds, random_state=seed, shuffle=True)

# For Linear regression, we can use explained variance value to evaluate the model

scoring = 'explained_variance'

# Train the model and run K-foLd cross-validation to validate/evaluate the model

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Print out the evaluation results
# Result: the average of all the results obtained from the k-fold cross validation

print("Average of all results from the K-fold Cross Validation, using exlpained variance:",
      results.mean())


# #### <div class="alert alert-info"> The closer the value is to 1, the better.  This validation rating is also better than the homework (0.190) and my previous model (0.547).  I would prefer a stronger model in real life, however, it is better than the previously derived models, so I will take the win!</div>

# In[ ]:




