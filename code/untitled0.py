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


# %%

def cleansing_data(file_loc):
    dataset = pd.read_csv(file_loc, decimal='.', thousands=',', encoding='utf-8')
    dataset = dataset.loc[:, ['id', 'latitude', 'longitude',
                                'neighbourhood_cleansed', 'room_type',
                                'accommodates', 'bedrooms', 'beds',
                                'price', 'minimum_nights', 'number_of_reviews',
                                'reviews_per_month', 'estimated_occupancy_l365d',
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
    
    return dataset


# %% Cleansing data for Jun

data_jun = cleansing_data("listings_jun.csv")
data_sept = cleansing_data("listings_september.csv")
    
print(data_jun.dtypes)