#!/usr/bin/env python
# coding: utf-8

# # Final Supervised ML:  Decision Tree Regression
# ## Kathleen Harris

# In[1]:


# Import Python Libraries: NumPy and Pandas

import pandas as pd
import numpy as np

# Import Libraries & modules for data visualization

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import scit-Learn module for the algorithm/model: DecisionTreeRegressor

from sklearn. tree import DecisionTreeRegressor

# Import scikit-Learn module to split the dataset into train/ test sub-datasets

from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ## 1) Load Data

# In[2]:


# Specify location of the dataset.

housingfile = './data/housing_boston.csv'


# In[3]:


# Load the data into a Pandas DataFrame

df= pd.read_csv (housingfile, header=None)


# In[4]:


# Specify the fields with their names

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
'TAX', 'PTRATIO', 'AA','LSTAT', 'MEDV']

# Load the data into a Pandas DataFrame

df = pd.read_csv(housingfile, names=names)


# In[5]:


# Look at the first five rows to verify it loaded properly
#  Look at the first 5 rows of data

df.head()


# ## 2) Preprocess Data
# ### Clean the data: Find and mark missing values

# In[6]:


df.isnull().sum()

# There are no missing data points


# In[7]:


# reducing the variables to the 5 selected in problem 1

df3= df[['INDUS','LSTAT','RM', 'AGE', 'MEDV']]


# In[8]:


# verify the the dataframe was properly defined
df3.head()


# ## 3) EDA

# In[9]:


# determine the number of records and variables
# preview the data types of all variables
print(df3.shape)
print(df3.dtypes)


# In[10]:


# View the summary statistics
print(df3.describe())


# ### Histogram - first visualization

# In[12]:


df3.hist(edgecolor= 'black',figsize=(14,12))
plt.show()


# ### Boxplot - second visualization

# In[13]:


df3.plot(kind="box", subplots=True, layout=(5,3), sharex=False, figsize=(20,18))
plt.show()


# ### Heatmap - third visualziation

# In[14]:


# Creating a heat map to better visualize the correlations

plt.figure(figsize =(16,10))
sns.heatmap(df3.corr(), annot=True)
plt.show()


# ## 4) Separate the Dataset into Input & Output NumPy Arrays

# In[15]:


# Store the dataframe values into a numPy array

array= df3.values

# Separate the array into input and output components by slicing
# Note we have additional variables now, so we need to include 
# an extra column!
# For X (input) [:,4] --> All the rows and columns from 0 up to 4

X = array [:, 0:4]

# For Y (output) [:4] --> All the rows in the last column (MEDV)

Y = array [:,4]


# ## 5) Spilt into Input/Output Array into Training/Testing Data

# In[16]:


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

# In[17]:


# Build the model

model = DecisionTreeRegressor(random_state=seed)

# Train the model using the training sub-dataset

model.fit(X_train,Y_train)

# Non-Linear --> NO coefficients and the intercept

DecisionTreeRegressor (criterion='mse', max_depth=None, max_features=None,
max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=2,
min_samples_split=2, min_weight_fraction_leaf=0.0 , random_state=seed,
splitter='best')


# ## 7) Calculate R Squared

# In[18]:


R_squared = model.score(X_test, Y_test)
print('R-Squared = ', R_squared)


# #### <div class="alert alert-info"> The R-squared value is 76.84%.  It is significantly stronger than the model from the homework which had an r-squared value of 24.95%.  It is also stronger than the linear regression model which had an r-squared value of 61.92%.  We have successfully strengthened the model from the homework by choosing stronger predictor variables and increased the overall strength of the new model with by using the regression tree algorithm. </div>

# ## 8/9) Prediction
# ### Predict the "Median value of owner-occupied homes in 1000 dollars"

# In[19]:


# First Record using averages found during EDA:
# LSTAT = 11.44
# RM = 6 (rounded to nearest whole number)
# INDUS = 10.3
# AGE = 65.56

model.predict([[11.44,6,10.3,65.56]])


# #### <div class="alert alert-info">Given the above parameters, the model predicts the median value of owner-occuped homes in the given suburb to be approximately $21,900.</div>

# In[20]:


# Second Prediction:
# LSTAT = 11
# RM = 4 (closer to min value)
# INDUS = 10
# AGE = 66

model.predict([[11,4,10,66]])


# #### <div class="alert alert-info">Interesting, changing the number of rooms to the minimum value did not change the prediction.  The median value of owner-occuped homes in the given suburb is still approximately $21,900.</div>

# ## 10) Evaluate the model using 10-fold cross validation

# In[21]:


# Evaluate the algorithm and Specify the K-size

num_folds = 10

# Fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated

seed = 7

# Split the whole data set into folds

kfold= KFold(n_splits=num_folds, random_state=seed, shuffle=True)

# For Linear regression, we can use MSE (mean squared error) value to evaluate 
# the model/algorithm

scoring = 'neg_mean_squared_error'

# Train the model and run K-foLd cross-validation to validate/evaluate the model

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Print out the evaluation results
# Result: the average of all the results obtained from the k-fold cross validation

print("Average of all results from the K-fold Cross Validation, using negative mean squared error:",
      results.mean())


# #### <div class="alert alert-info">This MSE, -24.87, is better than the MSE value from the homework (-64); thus, this model is an improvement from the homework.  The regression tree algorithm is also an improvement from the linear regression model which had a MSE of -32.61.</div>

# In[22]:


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


# #### <div class="alert alert-info">This value is closer to 1 than the linear regression model, whoo hoo! </div>

# In[ ]:




