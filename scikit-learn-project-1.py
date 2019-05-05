#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


# Related to scikit learn, Python machine learning module 
# to work with the built-in datasets
from sklearn import datasets 


# In[15]:


bh = datasets.load_boston()


# In[16]:


bh.keys()


# In[17]:


#print(bh.DESCR)
print(bh['DESCR'])


# In[18]:


bh.feature_names


# In[19]:


df=pd.DataFrame(data=bh.data,columns=bh.feature_names)
df['price']=bh.target
df.head()


# In[20]:


bh2 = datasets.load_breast_cancer()


# In[21]:


bh2.keys()


# In[22]:


bh2.feature_names


# In[23]:


df2=pd.DataFrame(data=bh2.data,columns=bh2.feature_names)
df2['price']=bh2.target
df2.head()


# In[ ]:




