#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


# Related to scikit learn, Python machine learning module 
# to work with the built-in datasets
from sklearn import datasets 


# # Boston Housing Market Data
# 

# In[26]:


bh = datasets.load_boston()


# In[27]:


bh.keys()


# In[28]:


#print(bh.DESCR)
print(bh['DESCR'])


# In[29]:


bh.feature_names


# In[30]:


df=pd.DataFrame(data=bh.data,columns=bh.feature_names)
df['price']=bh.target
df.head()


# # Breast Cancer Data

# In[31]:


bh2 = datasets.load_breast_cancer()


# In[ ]:





# In[32]:


bh2.keys()


# In[33]:


bh2.feature_names


# In[34]:


df2=pd.DataFrame(data=bh2.data,columns=bh2.feature_names)
df2['price']=bh2.target
df2.head()


# In[ ]:





# In[ ]:




