# Script for loading in all the time-series data

import pandas as pd
from os.path import join


def load_all_data_pd(path_train, path_test, files):
    """
    Load in all the time series data from all the specified files as Pandas Dataframes

    Parameters:
        path_train: (string) path to train folder
        path_test: (string) path to test folder
        files: (list) list containing all the file names (test and train file names must match)

    Returns:
        ((train_f1, test_f1), (train_f2, test_f2), ...): Tuple containing all the train and test files loaded.

    """

    all_data = []

    for file in files:  # Go through all the files

        data_train = pd.read_csv(join(path_train, file))  # Read in files as pandas dataframe
        data_test = pd.read_csv(join(path_test, file))

        all_data.append([data_train, data_test])

    all_data_t = tuple(map(tuple, all_data))

    return all_data_t
