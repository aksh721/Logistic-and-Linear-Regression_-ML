#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Model

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\House Price.csv")
#we have read the file House Price


# In[28]:


df.head()


# In[29]:


plt.plot(df['area'],df['price'],'-*')
plt.xlabel('Area')
plt.ylabel('Price')
#we have plot the graph for Area and Price here observe as area increases even Price increases


# In[30]:


#assigning the value for X and y with area and price values
X=df[['area']]
y=df[['price']]


# In[31]:


X


# In[32]:


y


# In[33]:


#here we train the LinearRegression model

from sklearn.linear_model import LinearRegression

LR_model=LinearRegression()
LR_model.fit(X,y)
print("Model trained sucessfully")


# In[34]:


LR_model.predict(X)


# In[35]:


plt.scatter(df['area'],df['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.plot(df['area'],LR_model.predict(X))


# In[36]:


LR_model.predict([[2600]])


# In[37]:


LR_model.predict([[80000]])


# In[25]:


LR_model.predict([[34000]])

