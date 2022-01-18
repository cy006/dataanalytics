#!/usr/bin/env python
# coding: utf-8

# # 2. Create clean Dataset

# ## Import dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Cleaning dataset

# In[2]:


df = pd.read_csv('/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Leasing_risk/Dataset1/train_u6lujuX_CVtuZ9i.csv')


# In[3]:


df.info()


# ### Removing features that have no informative value

# In[4]:


#Removing Loan_Status because a custom label is to be created later
df = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
df


# ### Remove missing values

# In[5]:


sns.heatmap(df.isnull(), cbar=False)
df.isna().sum()


# In[6]:


df.Gender.value_counts().plot(kind='bar')


# In[7]:


df.isna().sum()


# In[8]:


df


# In[9]:


# Usage of most_frequent strategy for categorical variable -> Gender
from sklearn.impute import SimpleImputer
x_string = df.Gender.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.Gender=imp.fit_transform(x_string)


# In[10]:


df.isna().sum()


# In[11]:


# Usage of most_frequent strategy for categorical variable -> Married
x_string = df.Married.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.Married=imp.fit_transform(x_string)


# In[12]:


df.isna().sum()


# In[13]:


# Usage of most_frequent strategy for categorical variable -> Dependents
x_string = df.Dependents.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.Dependents=imp.fit_transform(x_string)


# In[14]:


df.isna().sum()


# In[15]:


df.Self_Employed .value_counts().plot(kind='bar')


# In[16]:


# Usage of most_frequent strategy for categorical variable -> Self_Employed
x_string = df.Self_Employed.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.Self_Employed=imp.fit_transform(x_string)


# In[17]:


df.isna().sum()


# In[18]:


df.LoanAmount.hist(bins = 40)


# In[19]:


df.Loan_Amount_Term.value_counts().plot(kind='bar')


# In[20]:


# Usage of most_frequent strategy for categorical variable -> Loan_Amount_Term
x_string = df.Loan_Amount_Term.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.Loan_Amount_Term=imp.fit_transform(x_string)


# In[21]:


df.info()


# In[22]:


df.isna().sum()


# In[23]:


# Usage of most_frequent strategy for categorical variable -> LoanAmount
x_string = df.LoanAmount.to_numpy().reshape(-1, 1)
imp = SimpleImputer(missing_values=np.nan, strategy='median')
df.LoanAmount=imp.fit_transform(x_string)


# In[24]:


df.Credit_History.value_counts().plot(kind='bar')


# In[25]:


df =  df.drop('Credit_History', axis = 1)


# In[26]:


df.isna().sum()


# ## Creation of custom dataset for leasing risk usecase

# ### Shaping Loan amount

# In[27]:


df.LoanAmount.plot.hist(bins = 40)


# In[28]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*100.0 if x > 30.0 else x*3)


# In[29]:


df.LoanAmount  = df.LoanAmount.apply(lambda x: x*0.75 if x > 7000.0 else x*3)


# In[30]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*0.60 if x > 12000.0 else x*1)


# In[31]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*10.0 if x > 0.0 else x*1)


# In[32]:


df.LoanAmount  = df.LoanAmount.apply(lambda x: x*0.6 if x > 120000.0 else x*1)


# In[33]:


df.LoanAmount.plot.hist(bins = 40)


# In[34]:


df.LoanAmount  = df.LoanAmount.apply(lambda x: x*0.5 if x > 125000.0 else x*1)


# In[35]:


df.LoanAmount  = df.LoanAmount.apply(lambda x: x*17.0 if x < 40000.0 else x*1)


# In[36]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*2.0 if x < 30000.0 else x*1)


# In[37]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*1.4 if x > 50000.0 else x*1)


# In[38]:


df.LoanAmount.plot.hist(bins = 40)


# In[39]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*2.0 if x < 50000.0 else x*1)


# In[40]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*0.6 if x > 90000.0 else x*1)


# In[41]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*0.6 if x > 94000.0 else x*1)


# In[42]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*0.6 if x > 0.0 else x*1)


# In[43]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*0.9 if x > 0.0 else x*1)


# In[44]:


df.LoanAmount = df.LoanAmount.apply(lambda x: x*1.5 if x > 0.0 else x*1)


# In[45]:


df.LoanAmount.plot.hist(bins = 40)


# In[46]:


df['Fahrzeugwert'] = df.LoanAmount


# In[47]:


df.Loan_Amount_Term = df.Loan_Amount_Term.apply(lambda x: x/10)


# In[48]:


df.Loan_Amount_Term.value_counts().plot(kind='bar')


# In[49]:


df = df.loc[df['Loan_Amount_Term'] > 23.0]


# In[50]:


df.Loan_Amount_Term.value_counts().plot(kind='bar')


# In[51]:


df['leasing_monat_48M'] = df.Fahrzeugwert / 48
df['leasing_summe'] = df['leasing_monat_48M'] * df['Loan_Amount_Term']
df['leasingrate'] = (df['leasing_summe'] / df['Loan_Amount_Term'])/2
df['einkommen_025'] = df['ApplicantIncome'] * 0.25


# In[52]:


df = df.drop(['LoanAmount', 'leasing_summe', 'leasing_monat_48M'], axis=1)


# ## Set the right dtype for features

# In[53]:


df.info()


# In[54]:


df = df.astype({"Gender": 'category', "Married": 'category', "Dependents": 'category', "Education": 'category', "Self_Employed": 'category', "CoapplicantIncome": 'int64', "Loan_Amount_Term": 'int64', "Property_Area": 'category'})


# In[55]:


df.info()


# ## Create custom label with conditions

# In[56]:


df


# In[57]:


conditions = [
    (df['leasingrate'] > df['einkommen_025']) | (df['Self_Employed'] == 'Yes') & (df['Property_Area'] == 'Urban') & (df['Education'] == "Not Graduate"),
    (df['Loan_Amount_Term'] > 0)
    ]
                                         
values = ['Risk', 'No Risk']

# create a new column and use np.select to assign values to it using -> lists as arguments
df['label'] = np.select(conditions, values)

df.head(50)


# In[58]:


df.label.value_counts().plot(kind='bar')
df.label.value_counts()


# In[59]:


df.info()


# In[60]:


df.to_csv(r'/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Leasing_risk/Final_data/dataset_leasingrisk.csv', index = False)


# In[61]:


dfsub = df[['Gender', 'ApplicantIncome', 'leasingrate']]


# In[62]:


dfsub.to_csv(r'/Users/cenkyagkan/Desktop/OMM/7.Semester/Applied Data Analytics/Clustering/subdataset_customersegmentation.csv', index = False)


# In[ ]:




