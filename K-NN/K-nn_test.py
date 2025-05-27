# -*- coding: utf-8 -*-
"""
Created on Mon May 26 09:04:50 2025

@author: hindi
"""


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from imblearn import under_sampling, over_sampling
#from imblearn.over_sampling import SMOTE


train = pd.read_csv('train_60percent.csv')
#Initialisation du tableau
#On garde le nom des colonnes en mémoire
ligne0 = train.columns.tolist()

#%% Clean the data

# #Retirer les doublons et les valeurs manquantes
# train = train.drop_duplicates()
# train = train.dropna()

# #2eme version : remplacer par la médiane
# for i in range(len(ligne0)) : 
#     mediane = train[ligne0[i]].median()
#     train[ligne0[i]] = train[ligne0[i]].fillna(value = mediane)
    
    
# # #SMOTE
# X = train.drop('rainfall', axis=1)
# y = train['rainfall']
# # smote = SMOTE(random_state=42) # Set random_state for reproducibility
# # X_resampled, y_resampled = smote.fit_resample(X, y)
# # train_resampled = pd.DataFrame(X_resampled, columns=X.columns)
# # train_resampled['rainfall'] = y_resampled 

        
# #Normaliser 
# X = X.select_dtypes(include=['number'])

# norm_df=(train-train.mean())/train.std()



#%% Definition de mon modele
#tab = norm_df.values
tab = train.values

def Poids(tab):
    """
    Cette fonction nous permet de pondérer plus ou moins fortement les distances si il y a beaucoup de changements ou non dans une même colonne.
    Par exemple, on a remarqué que dans les 2 premières colonnes, les valeurs décimales ne varient pas alors le coef de ces colonnes sera faible
    On accorde ainsi plus d'importances aux distances plus faible, pour les remonter afin qu'elles soit proportionnelles à celles des colonnes changeant peu
    
    Parameters
    ----------
    tab : TYPE
        DESCRIPTION.
        le fichier de valeur de références

    Returns 
    -------
    poids : TYPE
        DESCRIPTION.
        une liste comportant le poids (=coefficient) à appliquer à la colonne i (où i : l'indice du coef dans la liste)

    """
    poids = []
    colonne = []
    for i in range(1,len(tab[0])-1): #car la derniere colonne contient la classe donc pas dans test
        for j in range(len(tab)) : 
            if tab[j][i] not in colonne : 
                colonne.append(tab[j][i])
        colonne_triee = sorted(colonne)
        #On prend l'indice correspondant à 5% de la longueur (par ex. sur 100 valeurs, le minimum sera la 5ième valeur et la maximum sera la 5ème avant-dernière valeur)
        imin = int(len(colonne_triee)*0.05)
        imax = int(len(colonne_triee)*0.05)
        min_consistant = colonne_triee[imin]
        max_consistant = colonne_triee[-imax]
        #On cherche l'écart entre la valeur max et la valeur min pour avoir une idée de la variation
        poids.append(1/(max_consistant-min_consistant)) 
        colonne=[]

    return poids

def kNN(donnee_a_tester, tableau=tab, k=13, poids = np.ones(12)):
    """
    Parameters
    ----------
    donnee_a_tester : LIST 
        DESCRIPTION.
    tableau : TAB, optional
        DESCRIPTION. The default is tab.
    k : INT, optional
        number of neighbors

    Returns
    -------
    proba : FLOAT
        probability that it's raining the next day.

    """
    distance={}
    #On va stocker les distances entre chaque valeur d'apprentissage et la valeur de test dans un dictionnaire : clé correspondant à l'indice de la valeur dans le tableau du fichier, valeur correspondant à la distance 
    for i in range(len(tableau)):
        somme = 0
        for j in range(1, len(tableau[0])-1):
            somme += ((tableau[i][j] - donnee_a_tester[j])*poids[j-1])**2
        distance[i] = math.sqrt(somme)
    #tri du dictionnaire selon leur valeur ordre croissant en conservant leur clé respective
    distance_triees = dict(sorted(distance.items(),key=lambda t: t[1]))
    indice_voisin = []
    for i in distance_triees.keys(): 
        indice_voisin.append(i)
    #liste pour conserver les indices des k plus proches voisins
    voisin_0=0
    voisin_1=0
    #on compte le nombre de voisins de chaque classe
    for i in range(k): 
        if(tableau[indice_voisin[i]][12]==0):
            voisin_0+=1
        elif(tableau[indice_voisin[i]][12]==1):
            voisin_1+=1
    proba = voisin_1/k
    return proba

#%% Fonction de test qui enregistre les résultats

def Test():
    """
    Fonction qui réalise le test du knn en écrasant le fichier sample_submission.csv pour y intégrer les résultats

    Returns
    -------
    None.

    """
    p = Poids(tab)
    fichier_test = pd.read_csv('train_40percent.csv')

    test = fichier_test.values
    repartition = [0,0]
    
    resultats = open('sample_submission.csv','w')
    resultats.write('Id,Label\n') 
    for ligne in test : 
        proba = kNN(ligne, tab, poids=p)
        if proba>=0.5 :
            label = 1
        else :
            label = 0
        repartition[label]+=1
        resultats.write(f'{ligne[0]},{proba}\n')
    resultats.close()
    print(repartition)
    