#!/usr/bin/env python
# coding: utf-8

# In[289]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[290]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='seaborn.axisgrid')


# In[291]:


ins_data_frame = pd.read_csv('./insurance.csv')

print(ins_data_frame.head())


# In[292]:


print(ins_data_frame.describe())


# In[293]:


print(ins_data_frame.info())


# Checking for Missing Values.

# In[294]:


print(ins_data_frame.isnull().sum())


# Data Pre-Processing
# 
# Encoding the categorical features

# In[295]:


# encoding 'sex' column
ins_data_frame.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
ins_data_frame.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
ins_data_frame.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[296]:


sns.displot(data=ins_data_frame, x="age",kind='kde')


# In[297]:


sns.countplot(ins_data_frame, x="sex")


# In[298]:


sns.displot(data=ins_data_frame, x="bmi",kind='kde')


# In[299]:


sns.countplot(ins_data_frame, x="children")


# In[300]:


sns.countplot(ins_data_frame, x="smoker")


# In[301]:


sns.countplot(ins_data_frame, x="region")


# In[302]:


sns.displot(data=ins_data_frame, x="charges",kde=True)


# Spliting the features and target from the Dataset

# In[303]:


X = ins_data_frame.drop(columns='charges', axis=1)
Y = ins_data_frame['charges']


# In[304]:


print(X.head())


# In[305]:


print(Y)


# In[306]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


# In[307]:


model = LinearRegression()


# In[308]:


model.fit(X_train,Y_train)


# In[309]:


train_data_pred = model.predict(X_train)


# In[310]:


r2_score_train = metrics.r2_score(Y_train,train_data_pred)


# In[311]:


print(r2_score_train)


# In[312]:


test_data_pred = model.predict(X_test)


# In[313]:


r2_score_test = metrics.r2_score(Y_test, test_data_pred)
print('R squared value : ', r2_score_test)


# In[314]:


input_data = (31,1,25.74,0,1,0)
column_names = ["age","sex","bmi","children","smoker","region"]
# changing input_data to a numpy array
input_np_arrray = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_np_arrray.reshape(1,-1)

prediction_value = model.predict(pd.DataFrame(data=input_data_reshaped,columns=column_names))

print(f"The insurance cost would be: ${round(prediction_value[0],2)}")

