#!/usr/bin/env python
# coding: utf-8

# # 3. EDA & Data Preparation

# ## Import Dependencies

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Data

# In[2]:


df = pd.read_csv('/Users/cenkyagkan/books/data_analytics/content/final_dataset_cluster.csv')


# In[3]:


df


# ## Explorative Datenanalyse

# In[4]:


df.info()


# - Nur bei Leasingrate gibt es Missing Values. Diese müssen entfernt werden, da sonst zu Problemen beim Anwenden eines Clusteralgorithmus kommen könnte.
# - Age, Einkommen und Leasingrate eignen sich für ein Clusterverfahren am besten, da es sich hierbei um numerische Werte handelt.

# In[5]:


df.describe()


# - Age: Der jüngste Leasingkunde ist 18 Jahre alt und der älteste 70 Jahre.
# - Einkommen: Die Jahresgehälter strecken sich von 15.000 bis zu 137.000 Euro.
# - Leasingrate: Die niedrigste Leasingrate beträgt 422 Euro und die höchste mit 1765 Euro.

# In[6]:


plt.figure(figsize=(10,6))
plt.title("Altersgruppen")
sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()


# - Man kann erkennen, dass die meisten Leasningkunden zwischen 30 und 40 Jahre alt sind.
# - Das Alter ist hierbei eine interessante Variable für die Kundensegmentierung.

# In[7]:


genders = df.Gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


# In[8]:


df.Gender.value_counts()


# - Interessant ist, dass mehr Frauen ein Fahrzeug leasen im Verlgeich zu den Männern.

# In[9]:


plt.figure(figsize=(10,6))
plt.title("Verteilung des Jahreskommens")
sns.axes_style("dark")
sns.histplot(data=df, x="Einkommen", kde=True);
plt.show()


# - Die meisten Kunden haben ein Jahreseinkommen zwischen 20.000 Euro und 80.000 Euro.
# - Es gibt aber auch ein paar 'Besserverdiener', die mehr als 80.000 Euro verdienen.

# In[10]:


plt.figure(figsize=(10,6))
plt.title("Verteilung des Jahreskommens")
sns.axes_style("dark")
sns.boxplot(data=df, x="Einkommen");
plt.show()


# In[11]:


df.Einkommen.median()


# - Aus dem Boxplot kann man erkennen, dass der Median etwas über 60.000 Euro Jahreseinkommen liegt.
# - Das erste Quantil gibt an, dass 25% der Kunden weniger als ca. 40.000 Euro verdienen.
# - Das dritte Quantil gibt an, dass 75% der Kunden weniger als ca. 80.000 Euro verdienen.
# - Des Weiteren kann man erkennen, dass es auch einen Ausreißerwert bei ca. 140.000 gibt.

# In[12]:


df[df['Einkommen'] > 130000]


# - Bei den Ausreißern handelt es sich hierbei um zwei Männer, die anfang 30 sind und die bei dieser Leasingrate vermutlich ein Sportwagen geleast haben. 

# In[13]:


plt.figure(figsize=(10,6))
plt.title("Verteilung der Leasingraten")
sns.axes_style("dark")
sns.histplot(data=df, x="Leasingrate", kde=True);
plt.show()


# In[14]:


plt.figure(figsize=(10,6))
plt.title("Verteilung der Leasingraten")
sns.axes_style("dark")
sns.boxplot(data=df, x="Leasingrate");
plt.show()


# - Man kann erkennen, dass die Leasingrate meistens zwischen knapp 600 Euro und 800 Euro liegt.
# - Des Weiteren gibt es auch ein paar Ausreißer, die eine Leasingrate über 1200 Euro haben.

# In[15]:


sns.pairplot(df);


# - Man kann aus dem Pairplot deutlich erkennen, dass zwischen dem Jahreseinkommen und der Leasingrate eine positive Beziehung vorliegt.
# - Dies kann man auch in der folgenden Korrelations-Heatmap erkennen.

# In[16]:


corr = df.corr()
matrix = np.triu(corr)
sns.heatmap(corr, mask = matrix, annot = True);


# - Nachdem wir nun die Daten besser verstanden haben, werden im folgenden die Daten für das Clustering vorbereitet

# ## Data Preparation

# ### Missing Values

# In[17]:


df.isna().sum()


# In[18]:


df[df['Leasingrate'].isna()]


# In[19]:


sns.set(rc={'figure.figsize':(10,8)})
sns.scatterplot(data=df, x="Einkommen", y="Leasingrate");


# Da wir aus der EDA erfahren haben, dass es eine Beziehung zwischen Einkommen und Leasingrate besteht, werde ich die MVs bei der Leasingsrate mit einer **Regressionsimputation** ersetzen.

# In[20]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

x = df[['Einkommen', 'Leasingrate']].to_numpy()

# Initialisierung des iterativen Imputers mit 10 Iterationen 
imp = IterativeImputer(max_iter=10, random_state=0)

# Fit und Transform für das Array x
imp.fit(x)
x_all_values = np.round(imp.transform(x), 0)


# In[21]:


df['Leasingrate'] = x_all_values[:, [1]]


# In[22]:


df.isna().sum()


# ### Z-Score-Standardisierung

# Da der K-Means distanzbasiert arbeitet, müssen die Variablen normalisiert werden, da sonst die Variable Einkommen im Vergleich zum Alter stärker "ins Gewicht" fallen könnte. So verwende ich an dieser Stelle, die Z-Score-Standardisierung, sodass die Variablen einen Mittelwert von 0 haben und eine Standardabweichung von 1. 

# In[23]:


from sklearn.preprocessing import StandardScaler

X = df[['Age', 'Einkommen', 'Leasingrate']].to_numpy()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[24]:


X_std


# In[ ]:




