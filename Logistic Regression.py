#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[9]:


#Log Reg : It is classsfication tech.
#It is used for binary(Two) class classfn and multi class classfn prob.


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df=pd.read_csv("C:\\Users\\AMAN BIRADAR\\Downloads\\Python\\insurance_data.csv")


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


plt.scatter(df['age'],df['bought_insurance'])
plt.xlabel("Age")
plt.ylabel("bought_insurance")


# In[17]:


# Split Dataset into training and testing

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['age']],df[['bought_insurance']],test_size=0.3,random_state=3)


# In[27]:


print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)

print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[28]:


from sklearn.linear_model import LogisticRegression
Log_model=LogisticRegression()
Log_model.fit(X_train,y_train)
print("Model trained Successfully...!!")


# In[29]:


y_pred=Log_model.predict(X_test)

y_pred


# In[30]:


from sklearn import metrics
acc = metrics.accuracy_score(y_pred,y_test)
print("Model Accuracy is : ",acc*100)


# In[31]:


Log_model.predict([[20]])


# In[32]:


Log_model.predict([[10]])

