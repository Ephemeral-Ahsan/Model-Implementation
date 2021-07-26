#!/usr/bin/env python
# coding: utf-8

# # Model Evaluation and Refinement

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


# Import clean data 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)


# In[3]:


df.to_csv('module_5_auto.csv')


# In[4]:


df.head()


# In[7]:


# let's use only numeric data from the dataset 'df'

df = df._get_numeric_data()
df.head()


# In[13]:


# Training and Testing

y_data = df['price']
x_data= df.drop(columns='price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.40, random_state=0)

print("number of x_test samples :", x_test.shape[0])
print("number of x_training samples:",x_train.shape[0])

print("number of y_test samples :", y_test.shape[0])
print("number of y_training samples:",y_train.shape[0])


# In[14]:


lre=LinearRegression()
lre.fit(x_train[['horsepower']],y_train)

lre.score(x_test[['horsepower']],y_test)


# #### Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation. Let's go over several methods that you can use for cross-validation.
# 

# # Cross-Validation Score
# Let's import model_selection from the module cross_val_score.

# In[15]:


Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
Rcross

print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())


# # Part 2: Overfitting, Underfitting and Model Selection¶
# It turns out that the test data, sometimes referred to as the "out of sample data", is a much better measure of how well your model performs in the real world. One reason for this is overfitting.
# 
# Let's go over some examples. It turns out these differences are more apparent in Multiple Linear Regression and Polynomial Regression so we will explore overfitting in that context.
# 
# Let's create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.

# In[17]:


# Polynomial transformations

pr=PolynomialFeatures(degree=2)
x_train_pr= pr.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr= pr.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])


# In[18]:


x_train_pr.shape


# In[20]:


# Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.

poly=LinearRegression()
poly.fit(x_train_pr,y_train)


# In[21]:


# Use the method "predict" to predict an output on the polynomial features, and
# then use the function "DistributionPlot" to display the distribution of the predicted test output vs. the actual test data.

yhat_test=poly.predict(x_test_pr)


# In[22]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# In[23]:


# "DistributionPlot" to display the distribution of the predicted test output vs. the actual test data.

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)


# # Ridge Regression and Grid Search
# 
# Ridge regression is a model tuning method that is used to analyse any data that suffers from multicollinearity.
# In this section, we will review Ridge Regression and see how the parameter alpha changes the model. Just a note, here our test data will be used as validation data.
# 
# The term alpha is a hyperparameter. Sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.

# In[24]:


from sklearn.model_selection import GridSearchCV

parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1


# In[28]:


# Ridge Regression Object

RR=Ridge()
RR

# Create a ridge grid search object:

Grid1 = GridSearchCV(RR, parameters1,cv=4, iid=None)


# In[31]:


Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

bestRR= Grid1.best_estimator_
bestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)


# In[ ]:




