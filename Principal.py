# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 02:14:07 2022

@author: orlan
"""

import pandas as pd
import re
import seaborn as sns

treino = pd.read_csv(r'C:\Users\orlan\Desktop\Kaggle\Getting_Started\Titanic\dataset\train.csv', sep = ',')

'''
891 linhas x 12 Colunas

PassengerId  - ID do passageiro
Survived     - Sobreviveu               - 0: Não / 1: Sim             - Variável Resposta
Pclass       - Ticket Class             - 1: 1st / 2: 2nd / 3: 3rd    
Name         - Nome passageiro
Sex          - Sexo passageiro          - male / female
Age          - Idade passageiro         - min: 0,42 / max: 80
SibSp        - Qtd de irmãos/Conjuges   - min: 0 / max: 5
Parch        - Qtd de pais/filho        - min: 0 / max: 6
Ticket       - Nº Ticket
Fare         - Valor da tarifa          - min: 0 / max: 512,329
Cabin        - Nº da cabine
Embarked     - Porto de Embarcação      - C: Cherbourg / Q: Queenstown / S: Southampton
'''


# Breve informação estatística
d = treino[['Age','SibSp','Parch','Fare']].describe().round(2)


# Informações nulas
info_nulas = treino.isna().sum()


# Retirar o pronome de tratamento
treino['pronome'] = treino['Name'].apply(lambda x: re.split("[.$]", x)[0])
treino['pronome'] = treino['pronome'].apply(lambda x: re.split(", ", x)[1])


# Porto de Embarcação vazia
orlando = treino[treino['Embarked'].isna()]



# Tratamento
# problema com a idade
orlando = treino[treino['Age'].isna()]
treino['Age2'] = (treino['Age'] - treino['Age'].mean()) / treino['Age'].std()
treino['Age3'] = (treino['Age'] - treino['Age'].min()) / (treino['Age'].max() - treino['Age'].min())

sns.histplot(data = treino, x = 'Age3')

'''
125 não sobreviveram
52 sobreviveram

124 H
53 F
'''





