
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from src.evaluation.scoring_functions import MASE, sMAPE
# import matplotlib2tikz


# Change paths if needed
correlations_lines = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "filtered_correlations.csv")).readlines()
raw_data_lines = open(os.path.join(os.pardir, os.pardir, "data", "full", "train", "Daily.csv")).readlines()
naive_lines = open(os.path.join(os.pardir, os.pardir, "forecasts", "single_methods", "naive.csv")).readlines()
test_lines = open(os.path.join(os.pardir, os.pardir, "data", "full", "test", "Daily.csv")).readlines()

result_file = open(os.path.join(os.pardir, os.pardir, "scores", "analysis", "scores.csv"), "w")

data = []
for i, line in enumerate(raw_data_lines[1:]):
    split_line = line.strip().strip(",").strip('"').split('","')
    assert i + 1 == int(split_line[0][1:])
    data.append(list(map(float, split_line[1:])))

naive = []
for i, line in enumerate(naive_lines[1:]):
    split_line = line.strip().strip(",").split(',')
    assert i + 1 == int(split_line[0][1:])
    naive.append(list(map(float, split_line[1:])))

test = []
for i, line in enumerate(test_lines[1:]):
    split_line = line.strip().strip(",").split(',')
    test.append(list(map(float, split_line)))

meta = pd.read_csv(os.path.join(os.pardir, os.pardir, "data", "M4Info.csv"))
subset = meta.loc[meta.SP == "Daily"]


def string_to_date(string):
    split = string.split()[0].split("-")
    return datetime.date(int(split[0]), int(split[1]), int(split[2]))


dates = []
for i, (idx, row) in enumerate(subset.iterrows()):
    dates.append(string_to_date(row.StartingDate))


result_predictions = {}
result_correlations = {}
result_prediction_from = {}
result_start_stop = {}
for id, line in enumerate(correlations_lines):
    split_line = line.strip().split(',')

    row_id_i = int(split_line[0])  # Row number in Daily.csv
    row_id_j = int(split_line[1])
    overlap_start_i = int(split_line[2])  # Index where overlap starts
    overlap_end_i = int(split_line[3])
    overlap_start_j = int(split_line[4])  # Index where overlap ends
    overlap_end_j = int(split_line[5])
    correlation = float(split_line[6])  # Correlation

    row_name_i = "D" + str(row_id_i + 1)
    row_name_j = "D" + str(row_id_j + 1)

    train_row_i = data[row_id_i]
    train_row_j = data[row_id_j]

    length_i = len(train_row_i)
    length_j = len(train_row_j)

    # Take overlapping parts
    overlap_i = np.array(data[row_id_i][overlap_start_i: overlap_end_i])
    overlap_j = np.array(data[row_id_j][overlap_start_j: overlap_end_j])

    # Scale according to overlap
    scaled_i = (data[row_id_i] - np.mean(overlap_i)) / np.std(overlap_i)
    scaled_j = (data[row_id_j] - np.mean(overlap_j)) / np.std(overlap_j)

    max_value = np.max((np.max(scaled_i), np.max(scaled_j)))
    min_value = np.min((np.min(scaled_i), np.min(scaled_j)))

    # Align time series
    # if overlap_start_i == 0:
    #     overlap_start = overlap_start_j
    #     time_i = np.arange(overlap_start_j, overlap_start_j + length_i)
    #     time_j = np.arange(0, length_j)
    #     plt.plot((overlap_start_j, overlap_start_j), (min_value, max_value))
    # else:
    #     overlap_start = overlap_start_i
    #     time_i = np.arange(0, length_i)
    #     time_j = np.arange(overlap_start_i, overlap_start_i + length_j)
    #     plt.plot((overlap_start_i, overlap_start_i), (min_value, max_value))
    #
    # plt.plot(time_i, scaled_i)
    # plt.plot(time_j, scaled_j)
    # plt.plot((time_i[-1], time_j[-1]), (scaled_i[-1], scaled_j[-1]), ".")  # red dot to denote the end of time series
    # plt.show()

    overlap_last_elems_i = overlap_i[-14:]
    overlap_last_elems_j = overlap_j[-14:]

    if overlap_end_i == length_i:
        forecast = np.array(data[row_id_j][overlap_end_j: overlap_end_j+14])
        forecast = (forecast - np.mean(overlap_last_elems_j)) / np.std(overlap_last_elems_j) * np.std(overlap_last_elems_i) + np.mean(overlap_last_elems_i)
        time_i = np.arange(0, length_i)
        time_f = np.arange(length_i, length_i+14)
        forecast_row_id = row_id_i
        other_row_id = row_id_j
        forecast_row_start_end = overlap_start_i, overlap_end_i
        other_row_start_end = overlap_start_j, overlap_end_j
        # plt.plot(time_i, train_row_i)
        # plt.plot(time_f, forecast)
    else:
        forecast = np.array(data[row_id_i][overlap_end_i: overlap_end_i+14])
        forecast = (forecast - np.mean(overlap_last_elems_i)) / np.std(overlap_last_elems_i) * np.std(overlap_last_elems_j) + np.mean(overlap_last_elems_j)
        time_j = np.arange(0, length_j)
        time_f = np.arange(length_j, length_j+14)
        forecast_row_id = row_id_j
        other_row_id = row_id_i
        forecast_row_start_end = overlap_start_j, overlap_end_j
        other_row_start_end = overlap_start_i, overlap_end_i
        # plt.plot(time_j, train_row_j)
        # plt.plot(time_f, forecast)

    if len(forecast) < 14:
        print("Too short")
    else:
        train_mase = MASE(np.array(data[forecast_row_id]), np.array(test[forecast_row_id]), np.array(forecast), 1)
        if forecast_row_id in result_predictions:
            result_predictions[forecast_row_id].append(forecast)
            result_correlations[forecast_row_id].append(correlation)
            result_prediction_from[forecast_row_id].append(other_row_id)
            result_start_stop[forecast_row_id].append((forecast_row_start_end, other_row_start_end))
        else:
            result_predictions[forecast_row_id] = [forecast]
            result_correlations[forecast_row_id] = [correlation]
            result_prediction_from[forecast_row_id] = [other_row_id]
            result_start_stop[forecast_row_id] = [(forecast_row_start_end, other_row_start_end)]

    # plt.show()

class Results():
    def __init__(self, name):
        self.name = name
        self.mases = []
        self.smapes = []
        self.naive_mases = []
        self.naive_smapes = []

    def add(self, naive_mase, naive_smape, mase, smape):
        self.mases.append(mase)
        self.smapes.append(smape)
        self.naive_mases.append(naive_mase)
        self.naive_smapes.append(naive_smape)

    def calculate_result(self):
        mean_mase = np.mean(self.mases)
        mean_smape = np.mean(self.smapes)
        mean_naive_mase = np.mean(self.naive_mases)
        mean_naive_smape = np.mean(self.naive_smapes)
        owa = np.mean((mean_mase / mean_naive_mase, mean_smape / mean_naive_smape))
        print(self.name)
        print(len(self.mases))
        print(mean_mase, mean_smape, owa)
        result_file.write(self.name + "," + str(len(self.mases)) + "," + str(mean_mase) + "," + str(mean_smape) + "," + str(owa) + "\n")


all_results = Results("All")
t1_results = Results("T1")
t2_results = Results("T2")
t3_results = Results("T3")
t4_results = Results("T4")
other = Results("Other")
using_future = Results("Future")
using_past = Results("Past")

save = False
plots = False
histo = False
histo_data = []

if save:
    result_forecasts_file = open(os.path.join(os.pardir, os.pardir, "forecasts", "analysis", "analysis_forecasts" + str(time.time()) + ".csv"), "w")
    result_correlations_file = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "analysis_correlations" + str(time.time()) + ".csv"), "w")
    result_forecasts_file.write("V1,,,,,,,,,,,,,,\n")

for i in range(4227):
    row_name = "D" + str(i + 1)
    if i in result_predictions:
        best_index = np.argmax(result_correlations[i])
        final_prediction = result_predictions[i][best_index]

        naive_mase = MASE(np.array(data[i]), np.array(test[i]), np.array(naive[i]), 1)
        naive_smape = sMAPE(np.array(test[i]), np.array(naive[i]))
        train_mase = MASE(np.array(data[i]), np.array(test[i]), np.array(final_prediction), 1)
        train_smape = sMAPE(np.array(test[i]), np.array(final_prediction))

        if result_correlations[i][best_index] > 0.995:
            (forecast_start, forecast_end), (other_start, other_end) = result_start_stop[i][best_index]
            if forecast_end - 1 - forecast_start >= 14:
                all_results.add(naive_mase, naive_smape, train_mase, train_smape)
                forecast_start_date = dates[i]
                other_start_date = dates[result_prediction_from[i][best_index]]
                is_t1 = False
                is_t2 = False
                is_t3 = False
                is_t4 = False
                if i == result_prediction_from[i][best_index]:
                    print("T1", end=" ")
                    t1_results.add(naive_mase, naive_smape, train_mase, train_smape)
                    is_t1 = True
                if result_prediction_from[i][best_index] in result_prediction_from and not is_t1:
                    best_index_for_other = np.argmax(result_correlations[result_prediction_from[i][best_index]])
                    if i == result_prediction_from[result_prediction_from[i][best_index]][best_index_for_other]:
                        print("T2", end=" ")
                        is_t2 = True
                        t2_results.add(naive_mase, naive_smape, train_mase, train_smape)
                if forecast_start_date+datetime.timedelta(days=forecast_start) == other_start_date+datetime.timedelta(days=other_start) and not is_t1 and not is_t2:  # abs(forecast_start - other_start) < 2:
                    print("T3", end=" ")
                    t3_results.add(naive_mase, naive_smape, train_mase, train_smape)
                    is_t3 = True
                if forecast_start_date+datetime.timedelta(days=forecast_start) != other_start_date+datetime.timedelta(days=other_start) and not is_t1 and not is_t2:
                    print("T4", end=" ")
                    t4_results.add(naive_mase, naive_smape, train_mase, train_smape)
                    is_t4 = True
                if not (is_t1 or is_t2 or is_t3 or is_t4):
                    other.add(naive_mase, naive_smape, train_mase, train_smape)
                if is_t1+is_t2+is_t3+is_t4 > 1:
                    print("assigned two categories!")
                # if i in result_prediction_from[i]:
                #     t3_results.add(naive_mase, naive_smape, train_mase, train_smape)
                # if i in result_prediction_from[i] and result_prediction_from[i][best_index] in result_prediction_from[i]:
                #     t4_results.add(naive_mase, naive_smape, train_mase, train_smape)
                if other_start_date + datetime.timedelta(days=other_end - 1) > forecast_start_date + datetime.timedelta(days=forecast_end - 1):
                    using_future.add(naive_mase, naive_smape, train_mase, train_smape)
                    print("future", end=" ")
                else:
                    using_past.add(naive_mase, naive_smape, train_mase, train_smape)
                    print("past", end=" ")
                if save:
                    result_forecasts_file.write(row_name + "," + ",".join(map(str, final_prediction)) + "\n")
                    result_correlations_file.write(row_name + "," + str(result_correlations[i][best_index]) + "\n")
                print(i, result_prediction_from[i][best_index], train_mase, train_smape)
                # if train_mase < 11 and plots and is_t2 and i in [446,163] and result_prediction_from[i][best_index] in [446,163]:
                #     length = len(data[i])
                #     time_i = np.arange(0, length)
                #     time_f = np.arange(length, length + 14)
                #     # plt.subplot(311)
                #     # plt.plot(time_i, data[i])
                #     # plt.plot(time_f, result_predictions[i][best_index])
                #     # plt.subplot(312)
                #     # other_part = data[result_prediction_from[i][best_index]]
                #     # forecast_part = data[i]
                #     # other_part_overlap = other_part[other_start:other_end]
                #     # forecast_part_overlap = forecast_part[forecast_start:forecast_end]
                #     # other_part = (other_part - np.mean(other_part_overlap)) / np.std(other_part_overlap)
                #     # forecast_part = (forecast_part - np.mean(forecast_part_overlap)) / np.std(forecast_part_overlap)
                #     # time_j = np.arange(forecast_start, forecast_start + len(other_part))
                #     # plt.plot(time_j, other_part)
                #     # plt.plot(time_i, forecast_part)
                #     # plt.subplot(313)
                #     # plt.plot(result_predictions[i][best_index])
                #     # plt.plot(test[i])
                #     other_part = data[result_prediction_from[i][best_index]]
                #     forecast_part = data[i]
                #     other_part_overlap = other_part[other_start:other_end]
                #     forecast_part_overlap = forecast_part[forecast_start:forecast_end]
                #     other_part = (other_part - np.mean(other_part_overlap)) / np.std(other_part_overlap)
                #     forecast_part = (forecast_part - np.mean(forecast_part_overlap)) / np.std(forecast_part_overlap)
                #     time_j = np.arange(forecast_start, forecast_start + len(other_part))
                #     plt.plot(time_j, other_part)
                #     plt.plot(time_i, forecast_part)
                #     matplotlib2tikz.save("T2_" + str(i) + ".tex")
                #     plt.show()
                if histo:
                    assert forecast_end - forecast_start == other_end - other_start
                    histo_data.append(forecast_end - 1 - forecast_start)
        else:
            print("nope")
            if save:
                result_forecasts_file.write(row_name + ",0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
                result_correlations_file.write(row_name + "," + "0" + "\n")
    else:
        if save:
            result_forecasts_file.write(row_name + ",0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
            result_correlations_file.write(row_name + "," + "0" + "\n")

if histo:
    print(len(histo_data))
    print(histo_data)
    print(max(histo_data))
    plt.hist(histo_data, bins=[i*100 for i in range(43)])
    # matplotlib2tikz.save("histogramm.tex")
    plt.show()

if save:
    result_forecasts_file.close()

result_file.write("Name,Count,MASE,sMAPE,OWA\n")
all_results.calculate_result()
t1_results.calculate_result()
t2_results.calculate_result()
t3_results.calculate_result()
t4_results.calculate_result()
# using_past.calculate_result()
# using_future.calculate_result()
# other.calculate_result()
