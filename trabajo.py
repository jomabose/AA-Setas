#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:49:36 2020

@author: ismael
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

semilla=0

def cargarDataset():
    # Cargamos los datos
    
    print("Cargando datos...")
    datos = np.array(pd.read_csv('datos/agaricus-lepiota.data'))
    
    # Elimino atributo 11 (missing values > 20%) 
    datos = np.delete(datos,11,1)
    
    enc = preprocessing.OrdinalEncoder()
    labenc = preprocessing.LabelEncoder()
    '''
    # Los ordena en orden alfabetico (e=0,p=1)    
    labenc.fit(['p','e'])
    
    input(datos)
    
    datos = np.insert(enc.fit_transform(datos[:,1:]),0,labenc.transform(datos[:,0]),axis=1).astype('float64')
    '''
    print(datos)
    datos = enc.fit_transform(datos).astype('float64')
    input(datos)
    
    print("Datos cargados!!")
    
    print("\nPreprocesando los datos...")    

    
    print("Datos preprocesados!!")
    
    return x,y,x_test,y_test
    
if __name__ == "__main__":
    x,y,x_test,y_test = cargarDataset()
    
    # Realizamos regresion lineal (minimo cuadrados)
    print("\nMínimos cuadrados")
    reg = linear_model.LinearRegression()
    score = cross_val_score(reg, x, y, cv=5,)
    print("Coeficiente de determinación 5 fold:", 1-score)
    print("Media coeficiente de determinación:", 1-score.mean())
    
    # Ajuste del modelo
    reg.fit(x,y)
    print("E_in:",1-reg.score(x,y))
    print("E_test:",1-reg.score(x_test,y_test))
    
    input("\n_________________________________________\nPulse intro para continuar...\n_________________________________________\n\n")
    
    # Realizamos regresion lineal (SGD)
    print("SGD")
    sgd = linear_model.SGDRegressor(penalty="l1",random_state=semilla)
    score = cross_val_score(sgd, x, y, cv=5)
    print("Coeficiente de determinación 5 fold:",1-score)
    print("Media coeficiente de determinación:", 1-score.mean())
    
    # Ajuste del modelo
    sgd.fit(x,y)
    print("E_in:",1-sgd.score(x,y))
    print("E_test:",1-sgd.score(x_test,y_test))
