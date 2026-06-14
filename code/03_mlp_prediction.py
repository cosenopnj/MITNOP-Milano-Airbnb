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
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# %% PRIPREMA PODATAKA - JUN
"""
    Ucitavanje podataka

"""
dataset_jun = pd.read_csv("listings_jun_clean.csv")
dataset_sep = pd.read_csv("listings_sept_clean.csv")
 
#print(dataset_jun.dtypes)

"""
  Pravljenje nove kolone annual_revenue, svaki oglas ima cenu po noci i procenjenu popunjenost u
danima godisnje, mnozimo ih i dobijamo godisnji prihod i ta kolona predstavlja y, tj ono sto 
mreza treba da nauci da predvidi

    Takodje izbacujemo vrednosti iz annual_revenue koje su jednake 0 (jednake su 0 jer occupancy 
                                                                  kolona ima 0 vrednsoti te nam
                                                                  to nije od koristi) 
    
    Na osnovu kolinearne matrice odredjujem koje kolone da izbacim a koje da zadrzim i upotrebim
unutar neuronske mreze.

    
"""

dataset_jun['annual_revenue'] = dataset_jun['price'] * dataset_jun['occupancy']
dataset_jun = dataset_jun[dataset_jun['annual_revenue'] > 0].reset_index(drop = True)


# korelaciona matrica samo za numericke kolone
numeric_cols = dataset_jun.select_dtypes(include=[np.number]).columns
corr_matrix = dataset_jun[numeric_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Korelaciona matrica')
plt.tight_layout()
plt.show()

# razdvajanje y i x i izbacivanje kolona koje nisu relevantne za predvidjanje godisnjeg prihoda
y_jun = dataset_jun['annual_revenue']
x_jun = dataset_jun.drop(columns=[
    'id',
    'annual_revenue',
    'price_tier',
    'occupancy',
    'latitude',
    'longitude',
    'beds',
    'score_cleanliness',
    'score_communication',
    'score_value',
    'score_accuracy',
    'revenue'
])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(y_jun, bins=50)
plt.title('Distribucija prihoda - originalna')
plt.xlabel('Godišnji prihod (€)')
plt.subplot(1, 2, 2)
plt.hist(np.log1p(y_jun), bins=50)
plt.title('Distribucija prihoda - log')
plt.xlabel('log(prihod)')
plt.tight_layout()
plt.show()

# one-hot encoding koji pretvara tekstualne kolone u brojeve 
x_jun = pd.get_dummies(x_jun, columns=['room_type', 'neighbourhood'], drop_first=False)

# zamena nula u score kolonama medijanom, jer 0 nije validna ocena (ocene od 1 do 5)
x_jun['score_location'] = x_jun['score_location'].replace(0, x_jun['score_location'].median())
x_jun['score_checkin'] = x_jun['score_checkin'].replace(0, x_jun['score_checkin'].median())

#popunjavanje nedostajucih vrednosti
x_jun = x_jun.fillna(x_jun.median(numeric_only=True))

# razdvajanje podataka na test i validacioni skup
x_train, x_val, y_train, y_val = train_test_split(
    x_jun, y_jun, test_size=0.2, random_state=42
)

#skaliranje
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled   = scaler.transform(x_val)

#%% FORMIRANJE I TRENIRANJE NEURONSKE MREZE
# arhitektura 256 -> 128 -> 64 -> 1, dropout za regulaciju, ReLu aktivaciona funkcija
nm = Sequential()
nm.add(Input(shape=(x_train_scaled.shape[1],)))
nm.add(Dense(256, activation='relu'))
nm.add(Dropout(0.2))
nm.add(Dense(128, activation='relu'))
nm.add(Dropout(0.1))
nm.add(Dense(64, activation='relu'))
nm.add(Dense(1))

nm.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

# log transformacija y - da bih smanjio uticaj outliera tokom treniranja
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

#zaustavlja treniranje kada val_loss prestane da se poboljsava
early_stop = EarlyStopping(patience=20, restore_best_weights=True)

istorija = nm.fit(
    x_train_scaled, y_train_log,
    validation_data=(x_val_scaled, y_val_log),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

#cuvanje modela
nm.save('mlp_model.keras')


#%% EVALUACIJA - validacioni skup - jun
y_pred_log = nm.predict(x_val_scaled, verbose=0)
y_pred = np.expm1(y_pred_log)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\nRMSE: {rmse:.2f} €")
print()
print()


print(f"Finalni train loss: {istorija.history['loss'][-1]:.2f}")
print(f"Finalni val loss: {istorija.history['val_loss'][-1]:.2f}")
print()
print()

y_val_array = np.array(y_val)
y_pred_array = y_pred.ravel()

# RMSE samo za oglase ispod 50.000€
mask_50 = y_val_array < 50000
rmse_50 = np.sqrt(mean_squared_error(y_val_array[mask_50], y_pred_array[mask_50]))
print(f"RMSE za oglase ispod 50.000€: {rmse_50:.2f} €")
print(f"Broj tih oglasa: {mask_50.sum()}")
print()
print()

# RMSE samo za oglase ispod 20.000€
mask_20 = y_val_array < 20000
rmse_20 = np.sqrt(mean_squared_error(y_val_array[mask_20], y_pred_array[mask_20]))
print(f"RMSE za oglase ispod 20.000€: {rmse_20:.2f} €")
print(f"Broj tih oglasa: {mask_20.sum()}")
print()
print()

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
plt.xlabel('Stvarni prihod (€)')
plt.ylabel('Predviđeni prihod (€)')
plt.title('Predviđeni vs. Stvarni godišnji prihod')
plt.tight_layout()
plt.show()

# Grafikon train vs val loss
plt.figure(figsize=(10, 4))
plt.plot(istorija.history['loss'], label='train loss')
plt.plot(istorija.history['val_loss'], label='val loss')
plt.title('Train vs Val Loss')
plt.xlabel('Epoha')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %% EVALUACIJA - Test skup za septembar
dataset_sep['annual_revenue'] = dataset_sep['price'] * dataset_sep['occupancy']
dataset_sep = dataset_sep[dataset_sep['annual_revenue'] > 0].reset_index(drop=True)


y_sep = dataset_sep['annual_revenue']
x_sep = dataset_sep.drop(columns=[
    'id',
    'annual_revenue',
    'price_tier',
    'occupancy',
    'latitude',
    'longitude',
    'beds',
    'score_cleanliness',
    'score_communication',
    'score_value',
    'score_accuracy',
    'revenue'
])



x_sep = pd.get_dummies(x_sep, columns=['room_type', 'neighbourhood'], drop_first=False)
x_sep['score_location'] = x_sep['score_location'].replace(0, x_sep['score_location'].median())
x_sep['score_checkin'] = x_sep['score_checkin'].replace(0, x_sep['score_checkin'].median())
x_sep = x_sep.fillna(x_sep.median(numeric_only=True))
x_sep = x_sep.reindex(columns=x_jun.columns, fill_value=0)
x_sep_scaled = scaler.transform(x_sep)

y_pred_sep_log = nm.predict(x_sep_scaled, verbose=0)
y_pred_sep = np.expm1(y_pred_sep_log)

rmse_sep = np.sqrt(mean_squared_error(y_sep, y_pred_sep))
print(f"RMSE na septembru: {rmse_sep:.2f} €")
print()

y_sep_array = np.array(y_sep)
y_pred_sep_array = y_pred_sep.ravel()

mask_50_sep = y_sep_array < 50000
rmse_50_sep = np.sqrt(mean_squared_error(y_sep_array[mask_50_sep], y_pred_sep_array[mask_50_sep]))
print(f"RMSE na septembru ispod 50.000€: {rmse_50_sep:.2f} €")
print()
print()
mask_20_sep = y_sep_array < 20000
rmse_20_sep = np.sqrt(mean_squared_error(y_sep_array[mask_20_sep], y_pred_sep_array[mask_20_sep]))
print(f"RMSE na septembru ispod 20.000€: {rmse_20_sep:.2f} €")
print()
print()

plt.figure(figsize=(8, 6))
plt.scatter(y_sep, y_pred_sep, alpha=0.3, s=10)
plt.plot([y_sep.min(), y_sep.max()], [y_sep.min(), y_sep.max()], 'r--', linewidth=2)
plt.xlabel('Stvarni prihod (€)')
plt.ylabel('Predviđeni prihod (€)')
plt.title('Predviđeni vs. Stvarni godišnji prihod - Septembar')
plt.tight_layout()
plt.show()

# mala analiza podataka
# zasto postoji velika razlika u RMSE i sta na nju utice?
print(f"Najskuplja cena u junu: {dataset_jun['price'].max():.2f} €")
print(f"Najskuplja cena u septembru: {dataset_sep['price'].max():.2f} €")
print()
print()
print(f"Maksimalni prihod u junu: {dataset_jun['annual_revenue'].max():.2f} €")
print(f"Maksimalni prihod u septembru: {dataset_sep['annual_revenue'].max():.2f} €")
print()
print()
print(y_jun.describe())
print()
print(y_sep.describe())
print()
print(dataset_jun.loc[dataset_jun['price'].idxmax(), ['price', 'occupancy', 'annual_revenue', 'room_type', 'neighbourhood']])
print()
print(dataset_sep.loc[9175, ['price', 'occupancy', 'room_type', 'neighbourhood']])
print()
print()
print("y_jun: \n")
print(f"Oglasi iznad 100.000€: {(y_jun > 100000).sum()}")
print(f"Oglasi iznad 50.000€: {(y_jun > 50000).sum()}")
print(f"Oglasi ispod 50.000€: {(y_jun < 50000).sum()}")
print()
print("y_sep: \n")
print(f"\nOglasi iznad 100.000€: {(y_sep > 100000).sum()}")
print(f"Oglasi iznad 50.000€: {(y_sep > 50000).sum()}")
print(f"Oglasi ispod 50.000€: {(y_sep < 50000).sum()}")
print()
print()

