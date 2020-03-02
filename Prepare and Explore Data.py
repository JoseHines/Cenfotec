#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np


# In[4]:


#Carga los datos
datos = pd.read_csv("D:\Personal\Maestria\Cenfotec\Modulos\V - Data Science with Python\I - Get Started with Data Science and Python\default of credit card clients.csv", sep=',')

datos.columns


# In[5]:


##Vemos el contenido del dataset
datos.head()


# In[6]:


##Podemos ver un resumen de los datos contenidos por el dataset
datos.describe()


# In[7]:


datos.info()


# In[8]:


##Validamos los tipos de datos del set de datos
datos.dtypes


# In[9]:


#Removemos la columna ID
del datos['ID']

#Renombramos el nombre de la columna
datos = datos.rename(columns={'default payment next month': 'Default_Payment'})


# In[92]:


#datos["SEX"]= datos["SEX"].replace(1.000000, "Hombre") 
#datos["SEX"]= datos["SEX"].replace(2.000000, "Mujer") 
#datos["SEX"]
#datos["MARRIAGE"]= datos["MARRIAGE"].replace(1.000000, "Casado") 
#datos["MARRIAGE"]= datos["MARRIAGE"].replace(2.000000, "Soltero") 
#datos["MARRIAGE"]= datos["MARRIAGE"].replace(3.000000, "Divorsiado") 
#datos["MARRIAGE"]= datos["MARRIAGE"].replace(0.000000, "Otro") 
#datos["MARRIAGE"]


# In[218]:


convert_dict = {'SEX': str, 
                'EDUCATION': str,
                'MARRIAGE': str, 
                'PAY_0': str, 
                'PAY_2': str, 
                'PAY_3': str, 
                'PAY_4': str, 
                'PAY_5': str, 
                'PAY_6': str, 
                'Default_Payment': str 
               } 
  
datos = datos.astype(convert_dict)


# In[10]:


bin_labels = ['20-28', '28-34', '34-41', '41-79']

datos["CAT_AGE"] = pd.qcut(datos["AGE"], 4,  labels=bin_labels)


# In[11]:


print(datos.groupby("CAT_AGE").size())


# In[12]:


#Distribución de los datos para la variable objetivo
datos['Default_Payment'].value_counts().plot(kind='bar')


# In[239]:



g = sns.catplot('CAT_AGE', data=datos, hue='Default_Payment', kind='count', aspect=1.75)
g.set_xlabels('CAT_AGE')


# In[241]:


g = sns.catplot('SEX', data=datos, hue='Default_Payment', kind='count', aspect=1.75)
g.set_xlabels('SEX')


# In[240]:


datos.pivot_table('PAY_AMT1', 'SEX', 'PAY_0', aggfunc=len, margins=True)


# In[175]:


datos.pivot_table('PAY_AMT1', 'MARRIAGE', 'PAY_0', aggfunc=np.sum, margins=True)


# In[176]:


datos.pivot_table('PAY_AMT2', 'SEX', 'PAY_2', aggfunc=len, margins=True)


# In[15]:


#table = pd.crosstab(index=[datos.Default_Payment, datos.EDUCATION], columns=[datos.SEX,datos.MARRIAGE])
table = pd.crosstab(index=[datos.PAY_2, datos.EDUCATION], columns=[datos.Default_Payment])
table.unstack()


# In[179]:


#Distribución del limite
limit_dist = sns.distplot(datos['LIMIT_BAL'])
limit_dist.set_title("Distribución del limite")


# In[180]:


#Distribución de la Edad
age_dist = sns.distplot(datos['AGE'])
age_dist.set_title("Distribución de la Edad")


# In[182]:


#table = pd.crosstab(index=[datos.Default_Payment, datos.EDUCATION], columns=[datos.SEX,datos.MARRIAGE])
table = pd.crosstab(index=[datos.PAY_0, datos.SEX], columns=[datos.CAT_AGE])

table.unstack()


# In[183]:


#table = pd.crosstab(index=[datos.Default_Payment, datos.EDUCATION], columns=[datos.SEX,datos.MARRIAGE])
table = pd.crosstab(index=[datos.PAY_2, datos.SEX], columns=[datos.CAT_AGE])

table.unstack()


# In[184]:


#table = pd.crosstab(index=[datos.Default_Payment, datos.EDUCATION], columns=[datos.SEX,datos.MARRIAGE])
table = pd.crosstab(index=[datos.PAY_3, datos.SEX], columns=[datos.CAT_AGE])

table.unstack()


# In[185]:


plt.scatter(datos['Default_Payment'],datos['PAY_AMT1'])


# In[225]:


plt.scatter(datos['BILL_AMT1'],datos['PAY_AMT1'])


# In[186]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(datos['BILL_AMT1'])


# In[16]:


datos.boxplot(column='LIMIT_BAL', by='CAT_AGE', rot=90)

# Display the plot
plt.show()


# In[188]:


datos.plot(kind='scatter', x='AGE', y='LIMIT_BAL', rot=70)
plt.show()


# In[189]:


datos['CAT_AGE'].value_counts().plot(kind='bar')


# In[190]:


datos['SEX'].value_counts().plot(kind='bar')


# In[192]:


g = sns.catplot('CAT_AGE', data=datos, hue='SEX', kind='count', aspect=1.75)
g.set_xlabels('CAT_AGE')


# In[49]:


datos['EDUCATION'].value_counts().plot(kind='bar')


# In[193]:


g = sns.catplot('EDUCATION', data=datos, hue='SEX', kind='count', aspect=1.75)
g.set_xlabels('EDUCATION')


# In[47]:


datos['MARRIAGE'].value_counts().plot(kind='bar')


# In[50]:


datos['AGE'].value_counts().plot(kind='bar')


# In[17]:


##Matriz de Correlation
corrMat = datos.corr()
print(corrMat)


# In[18]:



corrMat.style.background_gradient(cmap='coolwarm')


# In[19]:


##Matriz de Covariance
#You have already used correlation to understand the strength of relationships between any two variables, but how can we ascertain the impact on has on another? Covariance is often used to gauge the linear degree of change between two variables. Simply put, you can use covariance to measure how changes in one variable are associated with changes in a second variable. This will be very important when studying the impact various features might have on default rates so make sure you understand it fully. 
covMat = datos.cov()
print(covMat)


# In[20]:


covMat.style.background_gradient(cmap='coolwarm')


# In[212]:


dummy = pd.get_dummies(datos['EDUCATION'])
dummy.head()


# In[22]:


#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

from sklearn.preprocessing import StandardScaler

features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

featuresnum = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

featurescat = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'CAT_AGE']

# Separando las variables dependientes numericas
x = datos.loc[:, features].values


# Separando las variables dependientes numericas
xnum = datos.loc[:, featuresnum].values

# Separamos las variables dependientes categoricas
xcat = datos.loc[:, featurescat].values

# Separando la variable objetivo
y = datos.loc[:,['Default_Payment']].values

# Normalizando las variables dependientes
xnum = StandardScaler().fit_transform(xnum)
xnum


# In[23]:


datoscat = pd.DataFrame(data = xcat
             , columns = featurescat)


# In[24]:


from sklearn.decomposition import PCA

pca = PCA(n_components=6)

principalComponents = pca.fit_transform(xnum)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'PCA_6'])


# In[25]:


finalDf = pd.concat([principalDf, datoscat, datos[['Default_Payment']]], axis = 1)
finalDf


# In[26]:


finalDf.info()


# In[28]:


##Matriz de Correlation
corrMat = finalDf.corr()
print(corrMat)
corrMat.style.background_gradient(cmap='coolwarm')


# In[31]:


from Dora import Dora
dora = Dora()

dora = Dora(output = 'A', data = datos)
dora.data

