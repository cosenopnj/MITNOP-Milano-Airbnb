# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:46:51 2026

@author: cosenopnj
"""
# %% libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import folium

# %% data import

data_jun = pd.read_csv("listings_jun_clean.csv", thousands=',', decimal='.', encoding='utf-8')
data_sept = pd.read_csv("listings_sept_clean.csv", thousands=',', decimal='.', encoding='utf-8')


features = [
    'price', 'min_nights', 'reviews_count', 'reviews_monthly',
    'occupancy', 'availability', 'score_rating', 'score_cleanliness',
    'score_location', 'score_communication', 'score_value',
    'score_checkin', 'score_accuracy', 'latitude', 'longitude',
    'capacity', 'bedrooms', 'beds'
]
# %% data cleaning


data_jun = data_jun[features].dropna()
data_sept = data_sept[features].dropna()

print("Jun shape:", data_jun.shape)
print("Sept shape:", data_sept.shape)

# %% Price scaling

scaler = StandardScaler()

jun_scaled = scaler.fit_transform(data_jun)
sept_scaled = scaler.transform(data_sept)

print("Jun scaled shape:", jun_scaled.shape)
print("Primer prve vrste:", jun_scaled[0])

# %% train - validation split 

X_train, X_val = train_test_split(jun_scaled, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# %% Autoencoder ( 18 -> 12 -> 6 )

input_dim = 18

encoder_input = keras.Input(shape=(input_dim,))
x = layers.Dense(12, activation='relu')(encoder_input)
x = layers.Dropout(0.1)(x)
x = layers.Dense(6, activation='relu')(x)

decoded = layers.Dense(12, activation='relu')(x)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(encoder_input, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# %% Training

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(X_val, X_val),
    verbose=1
)

# %% Plot

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epoha')
plt.ylabel('MSE Loss')
plt.title('Trening autoenkodera')
plt.legend()
plt.tight_layout()
plt.show()

# %% Anomaly detection

jun_reconstructed = autoencoder.predict(jun_scaled)
jun_errors = np.mean(np.square(jun_scaled - jun_reconstructed), axis=1)

threshold = np.percentile(jun_errors, 95)
print(f"Prag anomalije: {threshold:.4f}")

jun_anomalies = jun_errors > threshold
print(f"Broj anomalija u junskom skupu: {jun_anomalies.sum()}")
print(f"Procenat anomalija: {jun_anomalies.mean()*100:.1f}%")

# %% Anomaly detection - septembar
sept_reconstructed = autoencoder.predict(sept_scaled)

sept_errors = np.mean(np.square(sept_scaled - sept_reconstructed), axis=1)

sept_anomalies = sept_errors > threshold
print(f"Broj anomalija u septembarskom skupu: {sept_anomalies.sum()}")
print(f"Procenat anomalija: {sept_anomalies.mean()*100:.1f}%")

# %% reconstruction error plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(jun_errors, bins=100, color='steelblue', edgecolor='none')
plt.axvline(threshold, color='red', linestyle='--', label=f'Prag: {threshold:.4f}')
plt.xlabel('Greška rekonstrukcije')
plt.ylabel('Broj oglasa')
plt.title('Jun 2025 — greška rekonstrukcije')
plt.xlim(0, 2)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(sept_errors, bins=100, color='orange', edgecolor='none')
plt.axvline(threshold, color='red', linestyle='--', label=f'Prag: {threshold:.4f}')
plt.xlabel('Greška rekonstrukcije')
plt.ylabel('Broj oglasa')
plt.title('Septembar 2025 — greška rekonstrukcije')
plt.xlim(0, 2)
plt.legend()

plt.tight_layout()
plt.show()

# %% Results to dataframe
data_jun_results = data_jun.copy()
data_jun_results['neighbourhood'] = pd.read_csv("listings_jun_clean.csv")['neighbourhood'].values
data_jun_results['reconstruction_error'] = jun_errors
data_jun_results['is_anomaly'] = jun_anomalies

data_sept_results = data_sept.copy()
data_sept_results['neighbourhood'] = pd.read_csv("listings_sept_clean.csv")['neighbourhood'].values
data_sept_results['reconstruction_error'] = sept_errors
data_sept_results['is_anomaly'] = sept_anomalies

# Pregled anomalnih oglasa
jun_anomaly_df = data_jun_results[data_jun_results['is_anomaly'] == True]
print("Top 10 anomalija po grešci rekonstrukcije:")
print(jun_anomaly_df.sort_values('reconstruction_error', ascending=False)[
    ['price', 'reviews_count', 'occupancy', 'score_rating', 'neighbourhood', 'reconstruction_error']
].head(10))

# %% Anomaly map - Jun

jun_original = pd.read_csv("listings_jun_clean.csv")
jun_original = jun_original.dropna(subset=features)
jun_original = jun_original.reset_index(drop=True)

jun_original['reconstruction_error'] = jun_errors
jun_original['is_anomaly'] = jun_anomalies

map = folium.Map(location=[45.4642, 9.1900], zoom_start=12)

normal = jun_original[jun_original['is_anomaly'] == False]
for _, row in normal.sample(2000).iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='steelblue',
        fill=True,
        fill_opacity=0.4,
        popup=f"Cena: {row['price']}€ | Ocena: {row['score_rating']}"
    ).add_to(map)

anomaly = jun_original[jun_original['is_anomaly'] == True]
for _, row in anomaly.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Cena: {row['price']}€ | Ocena: {row['score_rating']} | Greška: {row['reconstruction_error']:.2f}"
    ).add_to(map)

map.save("anomaly_map_jun.html")
print("Mapa sacuvana kao anomaly_map_jun.html")

# %% Anomaly map - September
sept_original = pd.read_csv("listings_sept_clean.csv")
sept_original = sept_original.dropna(subset=features)
sept_original = sept_original.reset_index(drop=True)

sept_original['reconstruction_error'] = sept_errors
sept_original['is_anomaly'] = sept_anomalies

mapa_sept = folium.Map(location=[45.4642, 9.1900], zoom_start=12)

normalni_sept = sept_original[sept_original['is_anomaly'] == False]
for _, row in normalni_sept.sample(2000).iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='steelblue',
        fill=True,
        fill_opacity=0.4,
        popup=f"Cena: {row['price']}€ | Ocena: {row['score_rating']}"
    ).add_to(mapa_sept)

anomalije_sept = sept_original[sept_original['is_anomaly'] == True]
for _, row in anomalije_sept.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Cena: {row['price']}€ | Ocena: {row['score_rating']} | Greška: {row['reconstruction_error']:.2f}"
    ).add_to(mapa_sept)

mapa_sept.save("anomaly_map_sept.html")
print("Mapa za septembar sacuvana.")
