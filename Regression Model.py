#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# In[2]:


path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# # Simple Linear Regression

# In[4]:


lm=LinearRegression()

lm.fit(df[['highway-mpg']],df['price'])

lm.predict(df[['highway-mpg']])


# In[6]:


print(lm.coef_)
print(lm.intercept_)


# # Multiple Linear Regression

# In[9]:


Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y= df['price']

lm1=LinearRegression()
lm1.fit(Z,y)

lm1.predict(Z)


# In[10]:


print(lm1.coef_)
print(lm1.intercept_)


# # Regression Plot
# When it comes to simple linear regression, an excellent way to visualize the fit of our model is by using regression plots.
# 
# This plot will show a combination of a scattered data points (a scatterplot), as well as the fitted linear regression line going through the data. This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation, as well as the direction (positive or negative correlation).

# In[13]:


#Let's compare this plot to the regression plot of "highway-mpg".

plt.figure(figsize=(15,12))
sns.regplot(x="highway-mpg", y="price", data=df);
plt.ylim(0,);


# In[14]:


#Let's compare this plot to the regression plot of "peak-rpm".

plt.figure(figsize=(15, 12))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,);


# In[16]:


# Let's look at the correlation between the predcitor varibales
df[["peak-rpm","highway-mpg","price"]].corr()


# # <h3>Residual Plot</h3>
# 
# <p>A good way to visualize the variance of the data is to use a residual plot.</p>
# 
# <p>What is a <b>residual</b>?</p>
# 
# <p>The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.</p>
# 
# <p>So what is a <b>residual plot</b>?</p>
# 
# <p>A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.</p>
# 
# <p>What do we pay attention to when looking at a residual plot?</p>
# 
# <p>We look at the spread of the residuals:</p>
# 
# <p>- If the points in a residual plot are <b>randomly spread out around the x-axis</b>, then a <b>linear model is appropriate</b> for the data.
# 
# Why is that? Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.</p>
# 

# In[17]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()


# In[20]:


# let's plot distribution plot for MLR

Y_hat = lm1.predict(Z)

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show();


# # Pipeline
# Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.

# In[24]:


input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
# 
# Then, we can normalize the data, perform a transform and fit the model simultaneously.

# In[26]:


pipe=Pipeline(input)

Z.astype('float')

pipe.fit(Z,y)


# In[29]:


ypipe=pipe.predict(Z)
print(len(ypipe))
ypipe


# In[38]:


# SLR

print(lm.score(df[['highway-mpg']],df['price']))
print(mean_squared_error(df['price'],lm.predict(df[['highway-mpg']])))


# In[40]:


# MLR

print(lm1.score(Z,df['price']))
print(mean_squared_error(df['price'],lm1.predict(Z)))


# In[47]:


# Polynomial Regression

Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe=Pipeline(Input)
pipe

X = df[['highway-mpg']].astype(float)
pipe.fit(X,y)

ypipe=pipe.predict(X)
ypipe[0:4]


# In[49]:


r_squared = r2_score(y, ypipe)
print('The R-square value is: ', r_squared)
print(mean_squared_error(df['price'],ypipe))


# In[ ]:




