#!/usr/bin/env python
# coding: utf-8

# # 3. Explorative Datenanalyse

# ## Import Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# ## Import Data

# In[2]:


df = pd.read_csv('/Users/cenkyagkan/books/mynewbook/content/dataset_leasingrisk.csv')


# In[3]:


df


# In[4]:


df.info()


# - Von den kategorischen Features müssen Datentypen angepasst werden.
# - Es gibt keine Missing Values

# ## Data preparation

# ### Change dtype

# In[5]:


df = df.astype({"Gender": 'category', "Married": 'category', "Dependents": 'category', "Education": 'category', "Self_Employed": 'category', "Loan_Amount_Term": 'int64', "Property_Area": 'category', "label": 'category'})


# In[6]:


df.info()


# In[7]:


# Entfernung von einkommen_025, da es nur bei der Datensatzerstellung benötigt wurde.
df = df.drop('einkommen_025', axis=1)


# ### Überprüfung der einzelnen Features

# In[8]:


cols = df.columns.to_list()


# In[9]:


for col in cols:
    print(col)
    print(df[col].value_counts())
    print('\n')


# - Es gibt keine Fehler bzw. Inkonsistenzen in den Daten.
# - Das Label "No Riks" ist überpräsentiert in Daten -> Somit handelt es sich um einen imbalanced dataset. So kann man dieser Stelle auf eine Sampling Methode zurückgreifen.
# - Um das Ungleichgewicht in den Daten zu beseitigen, kann man später die Methode SMOTE (Synthetic Minority Oversampling Technique) verwenden, um mehr Instanzen von der unterpräsentierten Gruppe zu erzeugen.

# In[10]:


sns.countplot(x="label", data=df)


# ### Deskriptive Statistik

# In[11]:


df.describe()


# **ApplicantIncome:**
# - 25% der Personen haben Einkommen unterhalb von 2873,50
# - 50% der Personen haben Einkommen unterhalb von 3815,00
# - 75% der Personen haben Einkommen unterhalb von 5771,50
# - Das durchschnittliche Gehalt liegt bei 5316,26
# - Auffällig ist, dass das niedrgiste Gehalt nur bei 150 liegt. -> Muss näher betrachtet werden
# 
# **CoapplicantIncome**
# - Der Zweitverdiener bzw. Lebenspartner hat ein geringes Einkommen im Vergleich zum Hauptverdiener.
# 
# **Loan_Amount_Term**
# - Im Durschnitt wird ein Fahrzeug für 36 Monate geleast.
# 
# **Fahrzeugwert und Leasingrat**
# - Durchschnittlich haben die Fahrzeuge einen Wert von 58340 und eine Leasingrate von 607.

# ### Verteilung der Features

# In[12]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
df_low_income = df[df['ApplicantIncome'] < 8000]
sns.histplot(x= 'ApplicantIncome', hue="label", data=df_low_income, bins=30);


# - Man kann deutlich erkennen, dass vor allem die Personen die einen Gehalt unter 3000 haben, die Leasingrate nicht bezahlen können und somit für Mercedes Benz ein Risiko darstellen.

# In[13]:


df[(df.ApplicantIncome > 3900) & (df.label == 'Risk')]


# - Bei den Personen, die ein Einkommen über 4000 haben und trotzdem als Risiko eingestuft wurden, handelt es sich um Selbstständige.

# In[14]:


df[df['ApplicantIncome'] < 1500].head()


# - Bei den Personen die weniger als 1500 verdienen, kann man erkennen, dass der Zweitverdiener dafür ein höheres Einkommen hat.

# In[15]:


df_normal_income = df[df['ApplicantIncome'] < 25000]
sns.pairplot(hue="label", data=df_normal_income);


# - Da der Datensatz mit einfachen Bedingungen erstellt wurde, kann man erkennen, das bei der Erstellung des Labels auch Noise entstanden ist, wie bei dem Plot unten links (ApplicantIncome / leasungrate)

# ### Entfernung von Noise

# In[16]:


df[(df.ApplicantIncome > 15000) & (df.leasingrate > 600) & (df.label == 'Risk')]


# In[17]:


df = df.drop([489])
df[(df.ApplicantIncome > 15000) & (df.leasingrate > 600) & (df.label == 'Risk')]


# In[18]:


df_normal_income = df[df['ApplicantIncome'] < 25000]
sns.pairplot(hue="label", data=df_normal_income);


# - Auch im Pairplot kann man erkennen, dass vor allem Personen mit einem geringem Einkommen ein Risiko darstellen.

# In[19]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
df_low_income = df[df['ApplicantIncome'] < 8000]
sns.histplot(x= 'ApplicantIncome', hue="label", data=df_low_income, bins=30);


# In[20]:


df[(df.ApplicantIncome > 3900) & (df.label == 'Risk')]


# ### Korrelation

# In[21]:


corr = df.corr()
matrix = np.triu(corr)
sns.heatmap(corr, mask = matrix, annot = True, square= True);


# - Es besteht keine Korrelation unter den Features
# - Außer bei Farhzeugwert und Leasingrate, da hier die Leasingrate auf Basis des Fahrzeugwertes berechnet wurde.
# - Auch bei der logistischen Regression sollte keine Multikollinearität vorliegen, dies gilt es noch zusätzlich mit dem Varianzinflationsfaktor zu prüfen.

# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Fahrzeugwert', 'leasingrate']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)


# - Da der Fahrzeugwert mit der Leasingrate korreliert, werde ich das Feature "Fahrzeugwert" beim Modelbulidng mal entfernen und mal mitaufnehmen.

# In[23]:


df


# In[24]:


# Export csv
df.to_csv("/Users/cenkyagkan/books/mynewbook/content/df_leasingrisk_final_clean"+ ".csv", index=False)


# In[ ]:




