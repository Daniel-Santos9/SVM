# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:12 2020

@author: Daniel Santos
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import collections

base = pd.read_csv('base_credit.csv')

# Pré-processamento base
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Divisão da base
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#Classificando com svm
classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#calculando métricas
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


collections.Counter(classe_teste)