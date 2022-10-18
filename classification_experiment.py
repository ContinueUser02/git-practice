#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[57]:


data1 = pd.read_csv("DS.csv")
data1


# In[28]:


data1.columns = ['날짜','요일','지역','음식']
data1


# In[29]:


data1['가격'] = np.random.randint(15000,40000,657592)
data1


# In[30]:


data1.describe()


# In[38]:


# 월요일 = 1, 화요일 = 2, ....... 일요일 = 7
data1["요일"].value_counts()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[51]:


result = data1
result.head(5)


# In[58]:


x = result["요일"]
y = result["가격"]
plt.title("요일과 가격간 상관관계")
plt.scatter(x,y)
plt.show()


# In[ ]:




