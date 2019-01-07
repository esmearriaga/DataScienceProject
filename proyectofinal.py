#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:03:04 2018

@author: esmeraldaarriaga
"""
#%%Paqueterias
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import sklearn.metrics as sk
#%%Descarga de datos
data=pd.read_csv('../data/Kaggle_Training_Dataset.csv',header=0)

#%%Reporte rapido
quick_report1 = pd.DataFrame(data.describe().transpose())
quick_report2= pd.DataFrame(data.describe(include=['object']).transpose())

#%%Limpieza de datos
#se elimina la primera columna porque es el indice
data=data.drop('sku',1)

#Se elimina valores nan porque es alrededor del 10% de la info
data=data[~np.array(data.lead_time.isnull())]
#data.isnull().values.any() #comprueba si hay mas nans en dataframe

#Se cambia los valores de no/si a 0 y 1, 1(YES) siendo producto retrasado
def replace_text(x,to_replace,replacement):
    try:
        x=x.replace(to_replace,replacement)
    except:
        pass
    return x

data=data.apply(replace_text,args=('No',0))
data=data.apply(replace_text,args=('Yes',1))
#%%Seleccion de entrenamiento y prueba
X=data.iloc[:,0:21]
y=data.iloc[:,21:22]

#Normalizar datos
X=pd.DataFrame(normalize(X))

#70% entrenamiento 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#%%Modelos predictivos
#Logistico
lreg = LogisticRegression(random_state = 0)
lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)
#Matriz de confusion
cm_lreg = confusion_matrix(y_test, y_pred)
#Metricas de desempeño
print('\tAccuracy: %1.3f'%accuracy_score(y_test,y_pred))
print('\tPrecision: %1.3f'%precision_score(y_test,y_pred))
print('\tRecall: %1.3f'%recall_score(y_test,y_pred))
print('\tF1: %1.3f'%f1_score(y_test,y_pred))
#Cross validation
scores_lreg=cross_val_score(lreg, X_train, y_train, cv=15)
inf_scor_lreg=pd.DataFrame([scores_lreg.mean(),scores_lreg.std()],index=['Mean','Standard dev'])
#%%
#SVM
#kernel='rbf', degree=2
clf=svm.SVC(kernel='rbf', gamma=10)
clf.fit(X_train.iloc[0:100000,], y_train.iloc[0:100000,])
y_pred = clf.predict(X_test)
cm_clf = confusion_matrix(y_test, y_pred)
#Metricas de desempeño
print('\tAccuracy: %1.3f'%accuracy_score(y_test,y_pred))
print('\tPrecision: %1.3f'%precision_score(y_test,y_pred))
print('\tRecall: %1.3f'%recall_score(y_test,y_pred))
print('\tF1: %1.3f'%f1_score(y_test,y_pred))

#Cross validation
scores_clf=cross_val_score(clf, X_train.iloc[0:100000,], y_train.iloc[0:100000,], cv=5)
inf_scor_clf=pd.DataFrame([scores_clf.mean(),scores_clf.std()],index=['Mean','Standard dev'])

#%%Eliminando variables
var=pd.DataFrame()
var['Varianza']=X.iloc[:,:].apply(lambda x: x.var())
var=var.sort_values(['Varianza'],ascending=[True])
var=(var.T)

X=X.drop([11,12,15,17,18,20],1)
X_train=X_train.drop([11,12,15,17,18,20],1)
X_test=X_test.drop([11,12,15,17,18,20],1)

var=pd.DataFrame()
var['Varianza']=X.iloc[:,:].apply(lambda x: x.var())
var=var.sort_values(['Varianza'],ascending=[True])
var=(var.T)

core=pd.DataFrame(X.corr())

X=X.drop([4,5,6,7,8,9],1)
X_train=X_train.drop([4,5,6,7,8,9],1)
X_test=X_test.drop([4,5,6,7,8,9],1)

core=pd.DataFrame(X.corr())
#%%Modelos predictivos
#Logistico
lreg = LogisticRegression(C=0.1)
lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)
#Matriz de confusion
cm_lreg = confusion_matrix(y_test, y_pred)
#Metricas de desempeño
print('\tAccuracy: %1.3f'%accuracy_score(y_test,y_pred))
print('\tPrecision: %1.3f'%precision_score(y_test,y_pred))
print('\tRecall: %1.3f'%recall_score(y_test,y_pred))
print('\tF1: %1.3f'%f1_score(y_test,y_pred))
#%%Cross validation
scores_lreg=cross_val_score(lreg, X_train, y_train, cv=15)
inf_scor_lreg=pd.DataFrame([scores_lreg.mean(),scores_lreg.std()],index=['Mean','Standard dev'])
#%%
#SVM
#kernel='rbf', degree=2
clf=svm.SVC(kernel='rbf', gamma=10)
clf.fit(X_train.iloc[0:90000,], y_train.iloc[0:90000,])
y_pred = clf.predict(X_test)
cm_clf = confusion_matrix(y_test, y_pred)
#Metricas de desempeño
print('\tAccuracy: %1.3f'%accuracy_score(y_test,y_pred))
print('\tPrecision: %1.3f'%precision_score(y_test,y_pred))
print('\tRecall: %1.3f'%recall_score(y_test,y_pred))
print('\tF1: %1.3f'%f1_score(y_test,y_pred))

#Cross validation
scores_clf=cross_val_score(clf, X_train.iloc[0:90000,], y_train.iloc[0:90000,], cv=5)
inf_scor_clf=pd.DataFrame([scores_clf.mean(),scores_clf.std()],index=['Mean','Standard dev'])

#%%Buscar el polinomio "óptimo"
#Como no se que polinomio me conviene, intento con varios, analizo y luego elijo
ngrado = 3 #Grado del polinomio
grados = np.arange(1,ngrado)
ACCURACY = np.zeros(grados.shape)
PRECISION = np.zeros(grados.shape)
RECALL = np.zeros(grados.shape)
F1 = np.zeros(grados.shape)
NUM_VARIABLES = np.zeros(grados.shape)
#%%Modelo de regresión lineal
for ngrado in grados:
    poly=PolynomialFeatures(ngrado)
    Xasterisco=poly.fit_transform(X) #es el x modificado, el que se le grega la fila de 1's
    logreg = linear_model.LogisticRegression(C=1)
    logreg.fit(Xasterisco,y) #Entrena el modelo
    Yg=logreg.predict(Xasterisco) #Sacar el "y" estimado
    #Guardar las variables en las matrices
    NUM_VARIABLES[ngrado-1] = len(logreg.coef_[0])
    ACCURACY[ngrado-1] = sk.accuracy_score(y,Yg) #Emparejamiento Simple
    PRECISION[ngrado-1] = sk.precision_score(y,Yg) #Precision
    RECALL[ngrado-1] = sk.recall_score(y,Yg) #Recall
    F1[ngrado-1] = sk.f1_score(y,Yg) #F1
#%%Visualizar los resultados
plt.plot(grados,ACCURACY)
plt.plot(grados,PRECISION)
plt.plot(grados,RECALL)
plt.plot(grados,F1)
plt.legend(('Accuracy','Precision','Recall','F1'))
plt.grid()
plt.show()
#%%Visualizar el grado de polinomio
plt.bar(grados,NUM_VARIABLES)
plt.title('Relación Grado-Parámetros')
plt.xlabel('Grado del Polinomio')
plt.ylabel('Número de Parámetros (w´s)')
plt.grid()
plt.show()
#(Por lo que se observa en las graficas la respuesta sería el poinomio de grado 4)
#%%Seleccionar el grado óptimo del análisis anterior
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xasterisco = poly.fit_transform(X)
logreg = linear_model.LogisticRegression(C=1)
logreg.fit(Xasterisco,y)
Yg = logreg.predict(Xasterisco)
sk.accuracy_score(y,Yg) #Porcentaje de acierto en total, y lo muestra en la terminal
#%% Anlaisiar los coeficientes  más significativos
W = logreg.coef_[0]
plt.bar(np.arange(len(W)),W)
plt.title('Relación Varaible-Valor del Parametro')
plt.xlabel('Número de Varible (x´s)')
plt.ylabel('Valor del Parámetro (w´s)')
plt.show()
#%%Anlaizar los coeficientes más significativos
W = logreg.coef_[0]
Wabs = np.abs(W)
umbral = 0.5 #umbral que indica que tan significante o insignificante es el valor de un parámetro
indx = Wabs>umbral
Xasterisco_seleccionada = Xasterisco[:,indx] #Sub matriz de x asterisco con las variables de los parametros significativos
plt.bar(np.arange(len(W[indx])),W[indx])
plt.title('Relación Varaible-Valor del Parametro Significativos')
plt.xlabel('Número de Varible (x´s)')
plt.ylabel('Valor del Parámetro (w´s)')
plt.show()
#%%Reentrenar el modelo con las variables seleccionadas
logreg_entrenada = linear_model.LogisticRegression(C=1)
logreg_entrenada.fit(Xasterisco_seleccionada,y)
Yg_entrenado = logreg_entrenada.predict(Xasterisco_seleccionada)
sk.accuracy_score(y,Yg_entrenado) #Porcentaje de acierto en total, y lo muestra en la terminal
diferencia = sk.accuracy_score(y,Yg) - sk.accuracy_score(y,Yg_entrenado)
print('la diferencia en porcentaje de aciertos del modelo entrenado y no entrenado es: ')
print(diferencia)
#Se observa que pese a tener menos variables, el porcentaje de accuracy score entrenado
#y el porcentahe de acierto sin entrenar, es el mismo. Es decir que con menos variables
#se llegó exactamente al mismo resultado. (con umbral de 0.5)
#se hace las metricasde desemepeño
cm_lreg = confusion_matrix(y,Yg_entrenado)

print('\tAccuracy: %1.3f'%accuracy_score(y,Yg_entrenado))
print('\tPrecision: %1.3f'%precision_score(y,Yg_entrenado))
print('\tRecall: %1.3f'%recall_score(y,Yg_entrenado))
print('\tF1: %1.3f'%f1_score(y,Yg_entrenado))

















