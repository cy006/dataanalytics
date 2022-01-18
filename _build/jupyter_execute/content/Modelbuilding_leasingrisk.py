#!/usr/bin/env python
# coding: utf-8

# # 4. Modelbuilding

# ## Import Dependencies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import Data

# In[2]:


df = pd.read_csv('/Users/cenkyagkan/books/mynewbook/content/df_leasingrisk_final_clean.csv')


# In[3]:


df.head()


# In[4]:


df = df.astype({"Gender": 'category', "Married": 'category', "Dependents": 'category', "Education": 'category', "Self_Employed": 'category', "Loan_Amount_Term": 'int64', "Property_Area": 'category', "label": 'category'})


# In[5]:


df.info()


# ## One-Hot Encoding

# - Es gibt zwar Algorithmen die direkt mit kategorialen Daten arbeiten können wie bspw. DecisionTrees, jedoch gibt es auch einige Algorithmen die nicht direkt mit kategorialen Variablen arbeiten können. So müssen die Daten in eine numerische Form überführt werden. An dieser Stelle habe ich mit pd.get_dummies die kategorialen Daten in numerische Daten umgewandelt.

# In[6]:


df = pd.get_dummies(df, drop_first=True)


# In[7]:


df = df.rename(columns={'Education_Not Graduate': 'Education_Not_Graduate', 'Dependents_3+': 'Dependents_3more'})


# In[8]:


df


# ## Modelbuilding with statsmodels

# In[9]:


model = smf.glm(formula = 'label_Risk ~ ApplicantIncome + CoapplicantIncome + Loan_Amount_Term + Fahrzeugwert + leasingrate + Gender_Male + Married_Yes + Dependents_1 + Dependents_2 + Dependents_3more + Education_Not_Graduate + Self_Employed_Yes + Property_Area_Semiurban + Property_Area_Urban' , data=df, family=sm.families.Binomial()).fit()


# In[10]:


print(model.summary())


# - Betrachtet man den p-Value der unabhängigen Variablen, dann kann man erkennen, dass nur die Features ApplicantIncome, Fahrzeugwert, Leasingrate und Education_Not_Graduate signifikant sind. Somit werde das Modell nochmals überarbeiten.

# In[11]:


print(model.aic)


# - Für das model1 habe ich den Akaike information criterion (AIC) berechnet, um später beurteilen zu können, ob die Qualität des Modells im Vergleich zum model2 abnimmt oder zunimmt.
# - Beim model2 entferne ich alle Features, die einen p-Value über 0.05 haben und berechne dann nochmals den AIC. 

# ### Updating Modell

# In[12]:


model2 = smf.glm(formula = 'label_Risk ~ ApplicantIncome + Fahrzeugwert + leasingrate + Education_Not_Graduate ' , data=df, family=sm.families.Binomial()).fit()


# In[13]:


print(model2.summary())


# In[14]:


print(model2.aic)


# - Der AIC-Wert von 145 zeigt, dass das model2 im Verlgleich zum model1, das bessere Modell ist. Somit werde ich mit dem model2 weiter fortfahren.

# ### Predictions

# In[15]:


# Predict and join probabilty to original dataframe
df['Probability_no'] = model2.predict()


# In[16]:


df


# - Um den optimalen Schwellenwert zu defnieren, werde ich im folgenden unterschiedliche Schwellenwerte testen. Hierzu verwende ich Schwellenwerte von 0.3 bis 0.7.

# In[17]:


# Use thresholds to discretize Probability
df['Threshold 0.3'] = np.where(df['Probability_no'] > 0.3, 'No', 'Yes')
df['Threshold 0.4'] = np.where(df['Probability_no'] > 0.4, 'No', 'Yes')
df['Threshold 0.5'] = np.where(df['Probability_no'] > 0.5, 'No', 'Yes')
df['Threshold 0.6'] = np.where(df['Probability_no'] > 0.6, 'No', 'Yes')
df['Threshold 0.7'] = np.where(df['Probability_no'] > 0.7, 'No', 'Yes')

df


# ### Confusionmatrix und Metriken

# In[18]:


def print_metrics(df, predicted):
    # Header
    print('-'*50)
    print(f'Metrics for: {predicted}\n')
    
    # Confusion Matrix
    y_actu = pd.Series(df['label_Risk'], name='Actual')
    y_pred = pd.Series(df[predicted], name='Predicted')
    df_conf = pd.crosstab(y_actu, y_pred)
    display(df_conf)
    
    # Confusion Matrix to variables:
    pop = df_conf.values.sum()
    tp = df_conf['Yes'][0]
    tn = df_conf['No'][1]
    fp = df_conf['No'][0]
    fn = df_conf['Yes'][1]
    
    # Metrics
    accuracy = (tp + tn) / pop
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1_score:.4f} \n')


# In[19]:


print_metrics(df, 'Threshold 0.3')
print_metrics(df, 'Threshold 0.4')
print_metrics(df, 'Threshold 0.5')
print_metrics(df, 'Threshold 0.6')
print_metrics(df, 'Threshold 0.7')


# ## Zusammenfassung von model2
# - Mit dem Generalized linear Modell konnte mit dem Schwellenwert von 0.4 der beste F-Score mit 99% berechnet werden.
# - Ich habe mich gegen SMOTE entschieden, da diese Methode auch ein Overfitting mit sich bringen kann und das Modell trotz einem unbalancierten Modell sehr gut performt.
# - Da es in diesem Usecase darum geht, das Risiko von Leasinganträgen zu reduzieren, bevorzuge ich den Recall als Evaluationsmetrik, da der Fokus mehr auf false-negative liegt.
# - Wenn es tatsächlich ein Risiko ist, aber es als kein Risiko vorhergesagt wurde, ensteht für Mercedes Benz ein Schaden und diesen gilt es zu vermeiden. -> Deshalb bevorzuge ich den Recall als Evaluationsmetrik.
# - **Das Modell wurde hier mit dem ganzen Datensatz gefittet. In den folgenden Zeilen werde ich Traings- und Testdaten erstellen und mit vielen unterschiedlichen Klassifikationsalgorithmen weitere Modelle erstellen.**

# ## Modelbuilding with sklearn

# ### Import Dependencies

# In[20]:


import time
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer


# ### Create Train- and Testdataset

# In[21]:


df_sk = df.drop(['Probability_no', 'Threshold 0.3', 'Threshold 0.4', 'Threshold 0.5', 'Threshold 0.6', 'Threshold 0.7'], axis=1)


# In[22]:


X = df_sk.drop('label_Risk', axis=1)
y = df_sk['label_Risk']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)


# Der Wert **random_state** stellt die Reproduzierbarkeit sicher, **shuffle** sorgt dafür, dass die Daten durcheinander gewürfelt werden, für den Fall, dass alle Risikofälle in den Daten zusammenliegen, und **stratify** stellt sicher, dass der Prozentsatz der Risikofälle in den Trainings- und Testdaten gleich ist.

# ### Baseline Models

# In[24]:


classifiers = {
    "LogisticRegression": LogisticRegression(solver = 'lbfgs', max_iter=1000),
    "LGBMClassifier": LGBMClassifier(),
    "XGBClassifier": XGBClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(3),    
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "GaussianNB": GaussianNB()
}


# Im folgenden wird auch eine K fold cross validation durchgeführt. Dadurch wird sichergestellt, dass jeder Fold den gleichen Anteil an positiven (Risiko) Klassen enthält.

# In[25]:


import warnings
warnings.filterwarnings('ignore')

df_models = pd.DataFrame(columns=['model', 'run_time', 'F1_mean'])

for key in classifiers:

    print('*',key)

    start_time = time.time()

    classifier = classifiers[key]
    model = classifier.fit(X_train, y_train)
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    
    scorer = make_scorer(f1_score)
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring=scorer)

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60,2)),
           'F1_mean': cv_scores.mean()
    }

    df_models = df_models.append(row, ignore_index=True)


# - Hier habe ich einen benutzerdefinierten Scorer mit make_scorer() erstellt, um den durchschnittlichen F1_Score in der Kreuzvalidierung für jedes Modell zu berechnen.
# - Die Ergebnisse werden dann in dem Dataframe df_models abgespeichert.

# In[26]:


df_models.sort_values(by='F1_mean')


# In[27]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
plot_order = df_models.groupby('model')['F1_mean'].sum().sort_values(ascending=False).index.values
clrs = ['grey' if (x < max(df_models.F1_mean)) else '#1DB5DA' for x in df_models.F1_mean]
sns.barplot(x='F1_mean', y='model', data= df_models, palette=clrs, order=plot_order);


# - **Das Modell LogisticRegression hat im Verlgeich zu den anderen am besten abgeschnitten** -> Mit einem druchschnittlichen F1 Score von 0,959.
# - Der XBgoost_Classifier hat ebenfalls einen sehr guten F1_Score erzielen können und ist auf dem zweiten Platz gelandet.
# - Die logistische Regression werde in den folgenden Zeilen genauer betrachten, indem ich dafür eine Confusionmatrix erstelle und weitere Metriken berechne.

# ### Bestes Modell: Logistic Regression

# In[28]:


df_result = pd.DataFrame(columns=['model', 'tp', 'tn', 'fp', 'fn', 'correct', 'incorrect',
                                  'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

classifier = LogisticRegression(solver = 'lbfgs', max_iter=1000)
model = classifier.fit(X_train, y_train)
y_pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


row = {'model': 'XGBClassifier without SMOTE',
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'correct': tp+tn,
        'incorrect': fp+fn,
        'accuracy': round(accuracy,3),
        'precision': round(precision,3),
        'recall': round(recall,3),
        'f1': round(f1,3),
        'roc_auc': round(roc_auc,3),
    }

df_result = df_result.append(row, ignore_index=True)
df_result.head()


# - Ob ein Leasingantrag genehmigt werde soll oder nicht, kann das Klassifikationsmodell mit einer Genauigkeit von 100% einschätzen. Somit könnte man das Risiko von Zahlungsausfällen von den Leasingnehmern reduzieren.
# - Das perfekte Ergebnis überrascht micht nicht, da das Modell die "einfache" Struktur verstanden hat, wie ich das Label für diesen Datensatz erzeugt habe.

# In[29]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
pred_proba = model.predict_proba(X_test)

df_ = pd.DataFrame({'y_test': y_test, 'y_pred': pred_proba[:,1] > .5})
cm = confusion_matrix(y_test, df_['y_pred'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)

disp.plot()
plt.show()


# In[ ]:





# In[ ]:




