# coding: utf-8

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

numpy2ri.activate()

import os
import numpy as np
import pandas as pd
import time
from collections import defaultdict

season_horizon = {"daily": (1, 14)}

data_folder_path = os.path.join(os.pardir, os.pardir, "data", "full", "train")
timestamp = str(time.time())
forecasts_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "single_methods", "ets_" + timestamp)


def load_all_data_pd(path_train, files):
    all_data = []
    for file in files:  # Go through all the files

        data_train = pd.read_csv(os.path.join(path_train, file))  # Read in files as pandas dataframe

        all_data.append(data_train)

    return all_data


files = ["Daily.csv"]
(train_da), = load_all_data_pd(data_folder_path, files)

all_data = {"daily": (train_da)}

forecast = importr('forecast')


# import warnings
# warnings.filterwarnings('ignore')


def train_model_arima(data, filler_cols=(1, 0)):
    '''
    Use all the data to train models for each row using given method

    Parameters:
        method: method used for model (i.e LinearRegression, MLPRegressor, RandomForestRegressor, etc)
        data: (dict) containing key: train, test
        filler_cols: (tuple) 2-element tuple containing number of filler cols for train and test
        multiplier: (int) how many seasons are taken into account for data sampling
        methods_kwargs: (dict) dictionary that contains keyword argument pairs for method using for training the model
        fit_kwargs: (dict) dictionary that contains keyword argument pairs for fitting method of the model.

    Returns:
        y_preds: (list) list containing all the predictions
    '''

    a = time.time()  # Starting time

    preds = defaultdict(list)
    mases = defaultdict(list)
    smapes = defaultdict(list)

    for freq, (train) in data.items():

        season, horizon = season_horizon[freq]  # Get seasonality and horizon (frequence)

        a1 = time.time()  # Time end

        # Get predictions one by one
        for i, (ix, item) in enumerate(train.iterrows()):
            # specific stuff
            train_row = np.array(item[filler_cols[0]:].dropna().values, dtype=np.float32)

            # Train model
            # https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima
            ro.globalenv["train"] = train_row
            fit = forecast.ets(ro.globalenv["train"])
            preds_tmp = forecast.forecast(fit, h=horizon)
            preds_treal = np.array(preds_tmp[1], dtype=np.float32)

            preds[freq].append(preds_treal)

        b1 = time.time()  # Time end
        total1 = b1 - a1
        print("%s: %.3f" % (freq, total1))

    # Time
    b = time.time()
    total = b - a
    print("Total time taken: ", total, "\n")  # How much time spent?

    return preds, mases, smapes


preds, mases, smapes = train_model_arima(all_data)


def export_preds(all_data, preds, outp_name):
    dfs = []

    for i, (freq, train) in enumerate(all_data.items()):
        len_t = len(train)
        temp_preds = preds[freq]
        df = pd.DataFrame(temp_preds)
        df.index = train['V1']
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all.to_csv(outp_name + ".csv", header=True)


export_preds(all_data, preds, forecasts_file_path)

