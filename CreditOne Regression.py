#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt

#estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#model metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

#cross validation
from sklearn.model_selection import train_test_split


# In[172]:


#data
rawData = pd.read_csv("D:\Personal\Maestria\Cenfotec\Modulos\V - Data Science with Python\I - Get Started with Data Science and Python\default of credit card clients.csv", sep=',')
#Removemos la columna ID
del rawData['ID']

#Renombramos el nombre de la columna
rawData = rawData.rename(columns={'default payment next month': 'Default_Payment'})

features = rawData.iloc[:,0:23]

rawData.head()


# In[2]:


#data
rawData = pd.read_csv('D:\Personal\Maestria\Cenfotec\Modulos\V - Data Science with Python\III - Build and Evaluate Models\default of credit card clients2.csv', header=0)

del rawData['Unnamed: 0']
del rawData['CAT_AGE']

features = rawData.iloc[:,0:18]

rawData.head()


# In[3]:


rawData.info()


# In[4]:


#dependent variable
depVar = rawData['Default_Payment']


# In[5]:


test_size = 0.30
seed = 1106
x_train, x_test = train_test_split(features, test_size= test_size, random_state=seed, shuffle=True)


# In[6]:


x_train.shape, x_test.shape


# In[7]:



y_train, y_test = train_test_split(depVar, test_size= test_size, random_state=seed, shuffle=True)


# In[8]:


y_train.shape, y_test.shape


# In[21]:


modelSVC = SVC()
modelRF = RandomForestClassifier()
modelGBM = GradientBoostingClassifier()


# In[22]:


#Random Forest
modelRF.fit(x_train,y_train)


# In[23]:


##Obtencion de metricas de medicion para Random Forest
print(cross_val_score(modelRF, x_train, y_train)) 
print(modelRF.score(x_train, y_train))


# In[24]:


#Support Vector Classifier
modelSVC.fit(x_train,y_train)


# In[25]:


##Obtencion de metricas de medicion para Support Vector Regression
print(cross_val_score(modelSVC, x_train, y_train)) 
print(modelSVC.score(x_train, y_train))


# In[26]:


#Gradient Boosting Models
modelGBM.fit(x_train,y_train)


# In[27]:


##Obtencion de metricas de medicion para linear Regression
print(cross_val_score(modelGBM, x_train, y_train)) 
print(modelGBM.score(x_train, y_train))


# In[246]:


#Make Predictions SVC
predictions = modelSVC.predict(x_test)


exactitud = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
sensibilidad = recall_score(y_test, predictions)

print('Exactitud: %.3f' % exactitud)
print('Precision: %.3f' % precision)
print('Sensibilidad: %.3f' % sensibilidad)


# In[247]:


#Make Predictions
predictions = modelRF.predict(x_test)


exactitud = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
sensibilidad = recall_score(y_test, predictions)

print('Exactitud: %.3f' % exactitud)
print('Precision: %.3f' % precision)
print('Sensibilidad: %.3f' % sensibilidad)


# In[248]:


#Make Predictions
predictions = modelGBM.predict(x_test)


exactitud = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
sensibilidad = recall_score(y_test, predictions)

print('Exactitud: %.3f' % exactitud)
print('Precision: %.3f' % precision)
print('Sensibilidad: %.3f' % sensibilidad)


# In[262]:


c=np.random.random(len(predictions))

plt.scatter(y_test, predictions, c=c, alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();


# In[28]:


##Modelos tuneados

modelSVC = SVC(C=1.0, cache_size=400, degree=5, gamma='auto_deprecated',
    verbose=True)
modelRF = RandomForestClassifier(bootstrap=True, max_depth=10, max_features='auto', 
                                 n_estimators=10, n_jobs=10)
modelGBM = GradientBoostingClassifier(max_depth=4, n_estimators=100,
                           verbose=1)


# In[29]:


#Random Forest
modelRF.fit(x_train,y_train)

##Obtencion de metricas de medicion para Random Forest
print(cross_val_score(modelRF, x_train, y_train)) 
print(modelRF.score(x_train, y_train))


# In[30]:


#Support Vector Classifier
modelSVC.fit(x_train,y_train)

##Obtencion de metricas de medicion para Support Vector Regression
print(cross_val_score(modelSVC, x_train, y_train)) 
print(modelSVC.score(x_train, y_train))


# In[31]:


#Gradient Boosting Models
modelGBM.fit(x_train,y_train)

##Obtencion de metricas de medicion para linear Regression
print(cross_val_score(modelGBM, x_train, y_train)) 
print(modelGBM.score(x_train, y_train))


# In[32]:


#Make Predictions
predictions = modelGBM.predict(x_test)


exactitud = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
sensibilidad = recall_score(y_test, predictions)

print('Exactitud: %.3f' % exactitud)
print('Precision: %.3f' % precision)
print('Sensibilidad: %.3f' % sensibilidad)


# In[ ]:




