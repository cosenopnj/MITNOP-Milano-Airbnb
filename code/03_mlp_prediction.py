# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:54:05 2026

@author: DELL
"""

#%% librarys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%

dataset_jun = pd.read_csv("listings_jun_clean.csv")
dataset_sep = pd.read_csv("listings_sept_clean.csv") 

print(dataset_jun.dtypes)

"""
pravljenje nove kolone annual_revenue, svaki oglas ima cenu po noci i procenjenu popunjenost u
danima godisnje, mnozimo ih i dobijamo godisnji prihod i ta kolona predstavlja y, tj ono sto 
mreza treba da nauci da predvidi
"""

dataset_jun['annual_revenue'] = dataset_jun['price'] * dataset_jun['occupancy']
print(dataset_jun.dtypes)

print(dataset_jun['annual_revenue'].head())
print(dataset_jun['annual_revenue'].tail())

# razdvajanje y i x i izbacivanje kolona koje nisu relevantne za predvidjanje godisnjeg prihoda
y_jun = dataset_jun['annual_revenue']
x_jun = dataset_jun.drop(columns = [
        'id',
        'annual_revenue',
        'price_tier'
    ])
print(x_jun.columns)


# one-hot encoding koji pretvara tekstualne kolone u brojeve 
x_jun = pd.get_dummies(x_jun, columns=['room_type', 'neighbourhood'], drop_first=False)

#print(x.columns.tolist())
print(x_jun['room_type_Hotel room'].tail())


#popunjavanje nedostajucih vrednosti
x = x_jun.fillna(x_jun.median(numeric_only=True))

x_train, x_val, y_train, y_val = train_test_split(
    x_jun, y_jun, test_size=0.2, random_state=42
)

#skaliranje
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled   = scaler.transform(x_val)




 





