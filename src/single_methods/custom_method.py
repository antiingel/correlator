
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from evaluation.evaluation import MASE, sMAPE
import time

forecasts_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "single_methods", "custom_method_" + str(time.time()) + ".csv")

path_train = os.path.join(os.pardir, os.pardir, "data", "holdout", "train", "Daily.csv")
path_test = os.path.join(os.pardir, os.pardir, "data", "holdout", "test" "Daily.csv")

train = {}
test = {}

train["Daily"] = pd.read_csv(path_train)
test["Daily"] = pd.read_csv(path_test)

# fix the IDs from training set
for key in test.keys():
    test[key] = pd.concat([train[key]['V1'], test[key]], axis=1)

full_train = os.path.join(os.pardir, os.pardir, "data", "full", "train", "Daily.csv")
train_f = {}
train_f["Daily"] = pd.read_csv(full_train)

plots_on = False  # to turn the plots on or off


def func_lin(x, a, b, c):
    return a * x + b


def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def my_sin(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset


def fit_trend(serie, func='lin'):
    tss = serie
    x = np.linspace(0, len(tss), len(tss))
    if func == 'exp':
        popt, pcov = curve_fit(func_exp, x, tss, maxfev=10000)
        if plots_on:
            plt.plot(x, func_exp(x, *popt), 'r-', label="Fitted Curve")
            plt.plot(serie)
            plt.title('exponential fitting to the trend')
    else:
        popt, pcov = curve_fit(func_lin, x, tss, maxfev=10000)
        if plots_on:
            plt.plot(x, func_lin(x, *popt), 'r-', label="Fitted Curve")
            plt.plot(serie)
            plt.title('linear fitting to the trend')
    if plots_on:
        plt.show()
    return popt


def fit_resid(data, freq):
    # N = 1000 # number of data points
    N = data.shape[0]
    t = np.linspace(0, 4 * np.pi, N)
    # data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

    guess_freq = freq
    guess_amplitude = 3 * np.std(data) / (2 ** 0.5)
    guess_phase = 0
    guess_offset = np.mean(data)

    p0 = [guess_freq, guess_amplitude,
          guess_phase, guess_offset]

    # now do the fit
    fit = curve_fit(my_sin, t, data, p0=p0, maxfev=10000)

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = my_sin(t, *p0)

    # recreate the fitted curve using the optimized parameters
    data_fit = my_sin(t, *fit[0])

    if plots_on:
        plt.plot(data, '.')
        plt.plot(data_fit, label='after fitting')
        plt.plot(data_first_guess, label='first guess')
        plt.title('fitting to the residuals')
        plt.legend()
        plt.show()

    fh_range = range(len(data), len(data) + fh)
    if plots_on:
        plt.plot(fh_range, data)
        plt.plot(fh_range, my_sin(fh_range, *fit[0]))
        plt.show()

    return fit[0]


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

    if masep == 0:
        return np.mean(abs(y_test - y_hat_test))
    else:
        return np.mean(abs(y_test - y_hat_test)) / masep


def fealess5(train, test=None, datasets=[], verbose=True, variation_pick={}, start_row=-1):
    """using linear, exponential and sinosoidal fittings on seasonal decomposition of the ts
    Args:
        train (dict): a dictionary datasets (i.e. Yearly) and timeseries of each (ie. Y1, .., Y23000)
        test (dict): if there is a hold-out dataset to measure the accuracy with
        datasets (dict): datasets to be consider - the list must contain the keys existing in train and test dictionaries
        verbose (bool): whether to print some of the debugging/informative information
        variation_pick (dict): based on performance on hold-out id of best prediction variation can be obtained...
                        which is useful for the final submission otherwise a default variation is chosen! (recommended)
        start_row (int): start prediction for this row onwards from
    Returns:
        preds (dict): predictions with each ts
        id_of_bests (dict): id of best prediction to be used for each ts
    """

    preds = {}  # the predictions itself
    id_of_bests = {}  # keep track of combination of predictions used
    if datasets:
        dsets = [d for d in train.keys() if d in datasets]
    else:
        dsets = [d for d in train.keys()]
    print("picked dataset(s) : ", dsets)
    print("starting row: ", start_row)
    try:
        for d in dsets:
            print(d)
            for i, row in train[d].iterrows():
                if i < start_row:
                    continue

                ts_id = row['V1']
                fh, freq = 14, 1
                if freq == 1:
                    freq = 2

                if verbose:
                    print(ts_id)
                    print(fh, freq)

                tr = row[1:].dropna().astype('float')
                if test is not None:
                    te = test[d].iloc[i, 1:]

                result = seasonal_decompose(tr.dropna().tolist(), model='add', freq=freq, two_sided=False)
                if plots_on:
                    result.plot()
                    pyplot.show()

                chop = fh
                resid = result.resid[-chop:]
                trend = result.trend[-chop:]
                resid[np.isnan(resid)] = 0
                trend[np.isnan(trend)] = 0

                # make forecast
                fh_range = range(chop, chop + fh)
                seasonal_pred = result.seasonal[-fh:]

                # fitting trend
                trend_coef = fit_trend(trend, func='lin')
                trend_pred = func_lin(fh_range, *trend_coef)

                resid_coef = fit_trend(resid, func='lin')
                resid_pred2 = func_lin(fh_range, *resid_coef)

                # calculate different possible combination of predictions
                trends = [trend_pred, result.trend[-fh:]]
                resids = [resid_pred2, result.resid[-fh:]]
                y_hats = []
                for ti, t in enumerate(trends):
                    for ri, r in enumerate(resids):
                        y_hats.append(t + r + seasonal_pred)

                # calculate error based on hold-out
                errors = []
                if test:
                    for yhat in y_hats:
                        errors.append(MASE(tr, te, yhat, 1))
                    id_of_bests[ts_id] = np.argmin(errors)

                # pick the prediction
                if errors:
                    yhat = y_hats[np.argmin(errors)]
                elif ts_id in variation_pick.keys():  # based on previous hold-out experience
                    yhat = y_hats[variation_pick[ts_id]]
                else:
                    print(ts_id, "Caution: better to pick based on variation_pick for final submission!")
                    yhat = y_hats[0]

                # store the prediction
                preds[str(row[0])] = yhat

                # plot the print final result
                if test is not None:
                    smape_err = sMAPE(te, yhat)
                    mase_err = MASE(tr[-fh:], te, yhat, 1)  # only for yearly freq=1
                    if verbose:
                        print("smape=%3.3f" % smape_err)
                        print("mase=%3.3f" % mase_err)

                if plots_on and test is not None:
                    plt.plot(fh_range, te)
                    plt.plot(fh_range, yhat)
                    plt.title('final prediction result')
                    plt.legend(['test', 'yhat'])
                    plt.show()

                print("#" * 10) if verbose else None
            # end of loop on the rows
        # end of loop on dataset
    except Exception as e:
        print("An error occurred on this row: ", ts_id)
        print("error msg: ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return (preds, id_of_bests)  # latest predictions so that the rest could be continued in another execution

    return (preds, id_of_bests)


preds_holdout, id_of_best_calc = fealess5(train=train, test=test, datasets=['Daily'], verbose=False)
preds_final, _ = fealess5(train=train_f, test=None, datasets=['Daily'], verbose=False, variation_pick=id_of_best_calc)

all_preds_final = pd.DataFrame.from_dict(preds_final, orient='index')
all_preds_final.to_csv(forecasts_file_path, float_format='%.f')
