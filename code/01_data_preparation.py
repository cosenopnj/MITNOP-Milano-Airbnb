# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:46:51 2026

@author: cosenopnj, RastkooP, TeodoraSovic
"""
# %% librarys

import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy import stats


# %%

def cleansing_data(file_loc, output_loc):
    dataset = pd.read_csv(file_loc, decimal='.', thousands=',', encoding='utf-8')
    dataset = dataset.loc[:, ['id', 'latitude', 'longitude',
                                'neighbourhood_cleansed', 'room_type',
                                'accommodates', 'bedrooms', 'beds',
                                'price', 'minimum_nights', 'number_of_reviews',
                                'reviews_per_month', 'estimated_occupancy_l365d',
                                'estimated_revenue_l365d',
                                'availability_365', 
                                'review_scores_rating',
                                'review_scores_cleanliness',
                                'review_scores_location',
                                'review_scores_communication',
                                'review_scores_value',
                                'review_scores_checkin',
                                'review_scores_accuracy']]
    
    dataset = dataset.rename(columns={
        'neighbourhood_cleansed':       'neighbourhood',
        'accommodates':                 'capacity',
        'minimum_nights':               'min_nights',
        'number_of_reviews':            'reviews_count',
        'reviews_per_month':            'reviews_monthly',
        'estimated_occupancy_l365d':    'occupancy',
        'estimated_revenue_l365d':      'revenue',
        'availability_365':             'availability',
        'review_scores_rating':         'score_rating',
        'review_scores_cleanliness':    'score_cleanliness',
        'review_scores_location':       'score_location',
        'review_scores_communication':  'score_communication',
        'review_scores_value':          'score_value',
        'review_scores_checkin':        'score_checkin',
        'review_scores_accuracy':       'score_accuracy',
    })
    
    dataset["price"] = dataset["price"].str.replace("$", "", regex = False)
    dataset["price"] = dataset["price"].str.replace(",", "").astype("float64")

    dataset.dropna(inplace = True)
    
    p33 = dataset["price"].quantile(0.33)
    p66 = dataset["price"].quantile(0.66)
    dataset["price_tier"] = pd.cut(
        dataset["price"],
        bins=[0, p33, p66, float("inf")],
        labels=["budget", "mid", "premium"]
    )
    
    dataset.to_csv(output_loc, index=False)
    
    return dataset


# %% Cleansing data for Jun

data_jun = cleansing_data("listings_jun.csv", "listings_jun_clean.csv")
data_sept = cleansing_data("listings_september.csv", "listings_sept_clean.csv")
    
print(data_jun.dtypes)
print()
print("Ocisceni dataset za jun:")
print()
print(data_jun["price_tier"].value_counts())
# %%EDA vizualizacije

#HISTOGRAM-raspodela cena po kvartovima
plt.close("all")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.hist(data_jun["price"], bins = 3000)
ax1.set_xlim(0, 600)
ax1.set_ylim(0, 8000)
ax1.set_title("Distribucija cena - Jun 2025")
ax1.set_xlabel("Cena (€)")
ax1.set_ylabel("Broj oglasa")

ax2.hist(data_sept["price"], bins = 3000)
ax2.set_xlim(0, 600)
ax2.set_ylim(0, 8000)
ax2.set_title("Distribucija cena - Septembar 2025")
ax2.set_xlabel("Cena (€)")
ax2.set_ylabel("Broj oglasa")
plt.show()

top20_jun = data_jun.groupby("neighbourhood")["price"].median().sort_values(ascending=False).head(20).index
data_top20_jun = data_jun[data_jun["neighbourhood"].isin(top20_jun)]

top20_sept = data_sept.groupby("neighbourhood")["price"].median().sort_values(ascending=False).head(20).index
data_top20_sept = data_sept[data_sept["neighbourhood"].isin(top20_sept)]

#BOKSPLOT-raspodela cena po kvartovima(zbog preglednosti izdvojeno 20 kvartova)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

data_top20_jun.boxplot(column="price", by="neighbourhood", ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_ylim(0, 500)
ax1.set_title("Distribucija cena po kvartovima - Jun 2025")
ax1.set_xlabel("Kvart")
ax1.set_ylabel("Cena (€)")

data_top20_sept.boxplot(column="price", by="neighbourhood", ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_ylim(0, 500)
ax2.set_title("Distribucija cena po kvartovima - Septembar 2025")
ax2.set_xlabel("Kvart")
ax2.set_ylabel("Cena (€)")

plt.tight_layout()
plt.show()

#KORELACIONA MATRICA
num_cols = ["price", "min_nights", "reviews_count", "reviews_monthly", "revenue",
            "occupancy", "availability", "score_rating", "score_cleanliness",
            "score_location", "score_communication", "score_value", 
            "score_checkin", "score_accuracy", "capacity", "bedrooms", "beds"]

corr_jun = data_jun[num_cols].corr()
corr_sept = data_sept[num_cols].corr()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 14))
sns.heatmap(corr_jun, annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
plt.title("Korelaciona matrica - Jun 2025")
ax1.set_title("Korelaciona matrica - Jun 2025")

sns.heatmap(corr_sept, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
ax2.set_title("Korelaciona matrica - Sept 2025")
plt.show()

# INTERAKTIVNA MAPA GUSTINE OGLASA

mapa_jun = folium.Map(location=[45.464, 9.190], zoom_start=12)
HeatMap(data_jun[["latitude", "longitude"]].values).add_to(mapa_jun)
mapa_jun.save("mapa_jun.html")

mapa_sept = folium.Map(location=[45.464, 9.190], zoom_start=12)
HeatMap(data_sept[["latitude", "longitude"]].values).add_to(mapa_sept)
mapa_sept.save("mapa_sept.html")

#t-test
t_stat, p_value = stats.ttest_ind(data_jun["price"], data_sept["price"])
print(f"t-statistika: {t_stat:.4f}")
print(f"p-vrednost: {p_value:.4f}")


print("VIZUELNO BOXPLOT:")
plt.figure(figsize=(8, 6))
plt.boxplot([data_jun["price"], data_sept["price"]], 
            tick_labels=["Jun 2025", "Septembar 2025"])
plt.ylim(0, 500)
plt.title("Poređenje cena - Jun vs Septembar 2025")
plt.ylabel("Cena (€)")
plt.show()

#bar chart - udeo tipova smeštaja po cenovnim rangovima

print("CHARTS:")

room_tier_jun = data_jun.groupby(["price_tier", "room_type"]).size().unstack(fill_value=0)
room_tier_jun_pct = room_tier_jun.div(room_tier_jun.sum(axis=1), axis=0) * 100

room_tier_sept = data_sept.groupby(["price_tier", "room_type"]).size().unstack(fill_value=0)
room_tier_sept_pct = room_tier_sept.div(room_tier_sept.sum(axis=1), axis=0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

room_tier_jun_pct.plot(kind="bar", ax=ax1)
ax1.set_title("Udeo tipova smestaja - Jun 2025")
ax1.set_xlabel("Cenovni rang")
ax1.set_ylabel("Udeo (%)")
ax1.set_xticklabels(["budget", "mid", "premium"], rotation=0)

room_tier_sept_pct.plot(kind="bar", ax=ax2)
ax2.set_title("Udeo tipova smestaja - Septembar 2025")
ax2.set_xlabel("Cenovni rang")
ax2.set_ylabel("Udeo (%)")
ax2.set_xticklabels(["budget", "mid", "premium"], rotation=0)

plt.tight_layout()
plt.show()















# %%
