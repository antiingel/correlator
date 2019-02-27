
import pandas as pd
import os
import numpy as np
from time import time


single_methods_path = os.path.join(os.pardir, os.pardir, "forecasts", "single_methods")
result_path = os.path.join(os.pardir, os.pardir, "forecasts", "ensemble", "ensemble.csv")


def get_files(path, extension=".csv"):
    """
    Get files from directory with extension

    Params:
        path (string) - path to folder with files
        extension (string) - extension of files that are loaded

    Returns: list with path to files
    """

    files = []

    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path, file))

    print(files)

    return files


def load_files(path, extension=".csv"):
    """
    First gets files from path and read them to dataframe (atm works only for csv)

    Params:
        path (string) - path to folder with files
        extension (string) - extension of files that are loaded

    Returns: list with dataframes
    """
    files = get_files(path, extension)

    dataframes = []

    for file in files:  # Go throught the files
        dataframes.append(pd.read_csv(file))  # Append csv-file to the list

    return dataframes


def get_dfs_by_freq(path, freq="M", extension=".csv"):
    """
    First gets files from path and read them to dataframe (atm works only for csv),
    filters the dataframes, leaving only rows with suitable frequency

    Params:
        path (string) - path to folder with files
        freq (string) - to filter certain frequency (to take all use ""), (one frequency or all)
        extension (string) - extension of files that are loaded

    Returns: list with dataframes
    """

    dfs = load_files(path)

    for i, df in enumerate(dfs):

        # if df.columns[0] != "V1":  # If first column is not real indices, might need to change this
        #     df = df.drop(df.columns[0], axis=1)

        df_temp = df.loc[df.iloc[:, 0].str.startswith(freq)]  # Take only necessary rows
        df_temp.set_index(keys=df_temp.columns[0], inplace=True)  # Set "V1" as index of DataFrame
        if freq != "":
            df_temp = df_temp.dropna(axis=1, inplace=False)  # Drop NA columns

        dfs[i] = df_temp

    return dfs


def ensemble(path, method, m_kwargs={}, freq="M", extension=".csv"):
    """
    Ensemble results from certain folder using input method.

    Params:
        path (string) - path to folder with files
        method - method for ensembling, method should be able to use np.ndarray of shape (nr_files, horizon) as input
        m_kwargs (dict) - extra arguments needed for function method
        freq (string) - frequency of data needed (i.e "" for all, "M" for monthly and so on)
        extension (string) - extension of files that are loaded

    Returns: Dataframe with ensembled results
    """

    t1 = time()

    dfs = get_dfs_by_freq(path, freq, extension)

    print("Data loaded: ", time() - t1)

    # Variables
    t2 = time()
    len_dfs = len(dfs)
    ens_array = np.zeros([*dfs[0].shape])
    print(ens_array.shape)

    for j, (idx, row) in enumerate(dfs[0].iterrows()):  # Go through the rows (TODO check if all the labels present?)

        temp_array = [np.asarray(row.values)]  # Add first row into matrix

        for i in range(1, len_dfs):  # Add all rows to matrix. First is taken into account already

            try:  # If key in dataframe
                row = np.asarray(dfs[i].loc[idx].values)
                if len(row):
                    temp_array.append(row)
            except KeyError:
                pass

        temp_array = np.asarray(temp_array)
        temp_array[temp_array < 0] = 0  # Replacing negative values with zeros

        new_row = method(temp_array, **m_kwargs)
        ens_array[j] = new_row

        if (j + 1) % 10000 == 0:
            print("Time taken for 10k rows:", time() - t2)

    res = pd.DataFrame(index=dfs[0].index, data=ens_array)  # Ensembled DataFrame
    print("Total time:", time() - t1)

    return res


res = ensemble(path=single_methods_path, method=np.median, m_kwargs={"axis": 0}, freq="D")
res.to_csv(result_path)
