#!/usr/bin/env python
# coding: utf-8

# # 2. Create Dataset for Clustering

# ## Import Dependencies

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import random


# ## Import Data

# In[2]:


df = pd.read_csv('/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Clustering/Mall_Customers.csv')


# In[3]:


df


# ## Create custom dataset

# In[4]:


df['Einkommen'] = df['Annual Income (k$)']*1000
df = df.drop(['Annual Income (k$)', 'Spending Score (1-100)', 'CustomerID'], axis = 1)


# In[5]:


df


# In[6]:


def rate(x):
    if x["Einkommen"] < 15001:
        return random.randint(380,440)
    elif x["Einkommen"] > 15001 and x["Einkommen"] < 25000:
        return random.randint(440,500)
    elif x["Einkommen"] >= 25000 and x["Einkommen"] < 50000:
        return random.randint(500,650)
    elif x["Einkommen"] > 50000 and x["Einkommen"] < 75000:
        return random.randint(650, 800)
    elif x["Einkommen"] > 75000 and x["Einkommen"] < 90000:
        return random.randint(800,950)
    elif x["Einkommen"] > 90000 and x["Einkommen"] < 120000:
        return random.randint(1000, 1300)
    elif x["Einkommen"] >= 120000:
        return random.randint(1300, 1800)
    


# In[7]:


df["Leasingrate"] = df.apply(lambda x: rate(x), axis=1)


# In[8]:


df.head(20)


# In[9]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(data=df, x="Einkommen", y="Leasingrate");


# In[10]:


df.to_csv(r'/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Clustering/final_dataset_cluster.csv', index = False)


# In[ ]:




