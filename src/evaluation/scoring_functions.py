# Evaluation measures for time series
# Using methods from https://github.com/M4Competition/M4-methods/blob/master/ML_benchmarks.py

import numpy as np


def sMAPE(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(100 * 2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b)))


def MASE(insample, y_test, y_hat_test, freq):
    """
    Calculates Mean Absolute Scaled Error
    :param insample: insample data  (training data)
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency (seasonality)
    :return: MASE
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    # Get seasonal difference, i.e January 2017 - Jan 2016, Feb 2017 - Feb 2016, etc
    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep


def OWA(MASE, sMAPE, naive_mase=1.685, naive_smape=15.201):
    """
    Overall Weighted Average of MASE and sMAPE using Naive2 scores

    Parameters:
        MASE: (float) mean absolute scaled error
        sMAPE: (float) mean absolute percentage error
        naive_mase: (float) MASE score of naive method
        naive_smape: (float) sMAPE score of naive method
    Returns:
        OWA: (float) owerall weighted average
    """

    return np.mean([MASE / naive_mase, sMAPE / naive_smape])  # Average of relative evaluation scores
