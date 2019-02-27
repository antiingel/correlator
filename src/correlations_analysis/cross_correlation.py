
import numpy as np
import numba
# import pandas as pd
import os
# from daily.src.load_data import load_all_data_pd
# import matplotlib.pyplot as plt

category = "Daily"
raw_data_dir = os.path.join(os.pardir, os.pardir, "data", "full")
path_train = os.path.join(raw_data_dir, "train")
path_test = os.path.join(raw_data_dir, "test")

# meta = pd.read_csv(os.path.join(os.pardir, "data", "raw", "M4Info.csv"))
# subset = meta.loc[meta.SP == "Daily"]

# (train, test), = load_all_data_pd(path_train, path_test, ["Daily.csv"])

train_data_lines = open(os.path.join(path_train, category + ".csv")).readlines()
train_data = []
for line in train_data_lines[1:]:
    train_data.append(np.array(line.strip("\n").strip(',').strip('"').split('","')[1:], dtype=np.float32))


@numba.jit(nopython=True)
def correlator(ts1, ts2):
    ts1_longer = len(ts1) >= len(ts2)
    correlations = []
    if not ts1_longer:
        ts1, ts2 = ts2, ts1
    for i in range(len(ts1) + len(ts2) - 1):
        if i < len(ts2):
            window1 = ts1[:i+1]
            window2 = ts2[len(ts2)-1-i:]
        elif i < len(ts1):
            window1 = ts1[i-len(ts2)+1:i+1]
            window2 = ts2
        else:
            window1 = ts1[i-len(ts2)+1:]
            window2 = ts2[:len(ts2)-(i - len(ts1) + 1)]
        window1 = window1 - np.mean(window1)
        window2 = window2 - np.mean(window2)
        var1 = np.dot(window1, window1)
        var2 = np.dot(window2, window2)
        if var1 == 0 and var2 == 0:
            correlations.append(1)
        elif var1 == 0 or var2 == 0:
            correlations.append(0)
        else:
            correlations.append(np.true_divide(np.dot(window1, window2), np.sqrt(var1 * var2)))
    if not ts1_longer:
        result = np.array(correlations[::-1])
    else:
        result = np.array(correlations)
    return result


overlap_counter = 0
unique_counter = 0
time_series = []

import time
now = time.time()
# results = []
file = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "unfiltered_correlations.csv"), "w")
for i in range(len(train_data)):
    for j in range(i, len(train_data)):
        result = correlator(train_data[i], train_data[j])
        good_indices = np.where(result > 0.995)

        correlation_nr = 0
        if len(good_indices) > 0:
            # file.write(str(i) + "," + str(j) + "," + ",".join(map(str, good_indices[0])) + "," + ",".join(map(str, result[good_indices])) + "\n")  # [13:-13]
            line = str(i) + "," + str(j) + "," + ",".join(map(str, good_indices[0])) + "," + ",".join(map(str, result[good_indices])) + "\n"
            split_line = line.strip().split(",")
            i = int(split_line[0])
            j = int(split_line[1])
            splitting_index = int(len(split_line[2:])/2)

            if split_line[2:splitting_index+2] == [""]:
                print("no correlations")
                continue
            if len(train_data[i]) < 10 or len(train_data[j]) < 10:
                print("Too short")
                continue

            indices = list(map(int, split_line[2:splitting_index+2]))
            correlations = list(map(float, split_line[splitting_index+2:]))
            prev_values = (None, None, None)
            while correlation_nr < len(correlations):
                correlation = correlations[correlation_nr]
                best_index = correlation_nr
                best_corr = correlation
                while True:
                    correlation_nr += 1
                    if correlation_nr < len(correlations) and indices[correlation_nr - 1] + 1 == indices[correlation_nr] and correlations[correlation_nr] > 0.995:# and correlations[correlation_nr - 1] <= correlations[correlation_nr]:
                        if best_corr < correlations[correlation_nr]:
                            best_index = correlation_nr
                            best_corr = correlations[correlation_nr]
                    else:
                        break
                index = indices[best_index]
                correlation = correlations[best_index]
                if correlation > 0.995:
                    len_i = len(train_data[i])
                    len_j = len(train_data[j])
                    if 13 < index < len_i+len_j-1 - 13:
                        start_i = len_j - 1
                        stop_i = len_j+len_i - 1
                        start_j = index
                        stop_j = index+len_j
                        time_i = np.arange(start_i, stop_i)
                        time_j = np.arange(start_j, stop_j)
                        start_overlap = start_i if start_j <= start_i else start_j
                        stop_overlap = stop_i if stop_i <= stop_j else stop_j
                        overlap_i = np.array(train_data[i][start_overlap-start_i:stop_overlap-start_i])
                        overlap_j = np.array(train_data[j][start_overlap-index:stop_overlap-index])
                        if (stop_i - stop_overlap) + (stop_j - stop_overlap) < 14:
                            overlap_counter += 1
                            continue
                        if len(np.unique(overlap_i)) <= 3 or len(np.unique(overlap_j)) <= 3:
                            unique_counter += 1
                            continue
                        # print(i, j, correlation)
                        # if stop_overlap == stop_i:
                        #     if i not in time_series:
                        #         time_series.append(i)
                        # else:
                        #      if j not in time_series:
                        #          time_series.append(j)
                        file.write(",".join(map(str, (i, j, start_overlap-start_i, stop_overlap-start_i, start_overlap-index, stop_overlap-index, correlation)))+"\n")
                    else:
                        pass
    print(time.time() - now)

# results = np.array(results)
# print(results.shape)
# np.save("cross_correlation.npy", results)


