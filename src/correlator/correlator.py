
import numpy as np
import os
from time import time

holdout = False

size = 0  # Sizes index
sizes = ["full"]

data_index = 1  # times index
times = ["Monthly", "Daily", "Hourly", "Quarterly", "Weekly", "Yearly"]

horis = [18, 14, 48, 8, 13, 6]

# peris = [12,  7, 24, 4, 4, 3]
peris = [9,  7, 24, 4, 6.5, 3]
# peris = [7, 7, 7, 7, 7, 6.5]

now = time()

std_constant = float("inf")
simulate_bug_2 = False
std_str = str(std_constant).replace(".", "").ljust(2, "0") + "_"
bug_str = "with_bug_2_" if simulate_bug_2 else ""

if holdout:
    path_to = os.path.join(os.pardir, os.pardir, "data", "holdout", "train", times[data_index] + ".csv")
else:
    path_to = os.path.join(os.pardir, os.pardir, "data", "full", "train", times[data_index] + ".csv")
correlations_file_name = os.path.join(os.pardir, os.pardir, "correlations", "correlator", "correlator_correlations_" + std_str + bug_str + str(time()) + ".csv")
forecasts_file_name = os.path.join(os.pardir, os.pardir, "forecasts", "correlator", "correlator_forecasts_" + std_str + bug_str + str(time()) + ".csv")


def predict(row, horizon, period, number_of_windows_for_row, all_data, all_data_index, corr_mat, rname, att, corr_matrix_index):
    x_2 = row
    x_2 = x_2[-int(2*period):]

    choice_inds = np.argsort(corr_mat[corr_matrix_index])
    # print(np.sort(corr_mat[corr_matrix_index]))
    # print(choice_inds)
    k = att
    while True:
        corr_matrix_index_2 = choice_inds[-k]
        if np.isnan(corr_mat[corr_matrix_index][choice_inds[0]]):   # No suitable indexes on row, assume constant window
            with open(correlations_file_name, "a") as ff:
                ff.write(str(rname) + "," + str(corr_mat[corr_matrix_index][corr_matrix_index_2]) + "," + "," + "," + "const" + "\n")
            return np.ones(horizon) * np.average(x_2)
        while np.isnan(corr_mat[corr_matrix_index][corr_matrix_index_2]) and k < len(choice_inds):
            k += 1
            corr_matrix_index_2 = choice_inds[-k]
        # print(k, corr_mat[corr_matrix_index][corr_matrix_index_2], corr_mat[corr_matrix_index][choice_inds[-(k-1)]])
        # print(k, len(choice_inds))
        # Find row of correlation
        row_ind = 0
        loc = 0
        while loc < corr_matrix_index_2:
            loc += number_of_windows_for_row[row_ind]
            row_ind += 1
        row_ind -= 1
        # Use row index to find the fake horizon data
        in_row = corr_matrix_index_2 - (loc - number_of_windows_for_row[row_ind]) + int(2 * period)
        tr_i = all_data[row_ind][in_row - int(2 * period):in_row]
        tr_j = all_data[row_ind][in_row:in_row + horizon]
        # print(k, row_ind, in_row)
        # print(corr_mat[corr_matrix_index][corr_matrix_index_2], np.corrcoef(tr_i, x_2))
        # input()

        prop_res = fit(tr_i, x_2, tr_j)
        # print(name, np.std(prop_res), np.std(tr_i), np.std(tr_j), np.std(x_2), all_data_index, row_ind, in_row)
        if k <= len(choice_inds):
            if simulate_bug_2 and (np.std(prop_res) > np.std(tr_i) * std_constant) or not simulate_bug_2 and (np.std(prop_res) > np.std(x_2) * std_constant):    # Too shaky to believe
                print("shaky", name, np.std(prop_res), np.std(tr_i), np.std(tr_j), np.std(x_2), all_data_index, row_ind, in_row)
                # with open(result_file_name, "a") as ff:
                #     ff.write(str(rname) + "," + str(corr_mat[corr_matrix_index][corr_matrix_index_2]) + "," + str(row_ind) + "," + str(in_row) + "," + "shaky" + "\n")
                # return prop_res
                # return predict(row, horizon, period, number_of_windows_for_row, all_data, row_ind, corr_mat, rname, k + 1, corr_matrix_index)
                k += 1
                continue
            else:
                with open(correlations_file_name, "a") as ff:
                    ff.write(str(rname) + "," + str(corr_mat[corr_matrix_index][corr_matrix_index_2]) + "," + str(row_ind) + "," + str(in_row) + "," + "ok" + "\n")
                return prop_res
        else:
            with open(correlations_file_name, "a") as ff:
                ff.write(str(rname) + "," + str(corr_mat[corr_matrix_index][corr_matrix_index_2]) + "," + "," + "," + "all_shaky" + "\n")
            return np.ones(horizon) * np.average(x_2)


def fit(row1, row2, res1):
    res1_a = np.array(res1, dtype=np.float64)
    avg1 = np.average(row1)
    std1 = np.std(row1)
    avg2 = np.average(row2)
    std2 = np.std(row2)
    if std1 == 0:
        # print("std1", std1, std2)
        res2 = res1_a - avg1 + avg2
    else:
        res2 = (res1_a - avg1) / std1 * std2 + avg2
    return list(res2)


def corr2_coeff(A, B):
    # https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:, None], ssB[None]))


def done(choice):
    doneset = set()
    # with open("predictions_corr5__"+str(sizes[size])+".csv") as fz:
    #     for row in fz:
    #         doneset.add(row.split(",")[0])
    return doneset


# Parse arguments
timez = data_index

f = open(forecasts_file_name, "a")
f.write("V1," + ",".join(map(str, [i for i in range(horis[data_index])])) + "\n")

doneset = done(sizes[size])
# Do work
results = []
row_endings = []
print("Doing", times[timez], "now")
data = open(path_to)
number_of_windows_for_row = []    # number of series gained from each row
windows = []
period = peris[timez]
horizon = horis[timez]
counter = 0
alldata = []
rownames = []

for data_row in data:
    if len(data_row) < 2:
        continue
    if counter % 200 == 0:
        print("at", counter, time()-now)
    counter += 1
    row = data_row.split(",")
    if row[0] == "V1" or row[0] == '"V1"':
        name_ind = 0
        continue
    else:
        if row[1] == "V1":
            name_ind = 1
            continue
    
    stop = 0
    if holdout:
        tr_data = np.array([float(num) for num in row[name_ind+1:] if num.strip() != ""])
    else:
        tr_data = np.array([float(num.strip()[1:-1]) for num in row[name_ind+1:] if num.strip()[1:-1] != ""])
    rownames.append(row[name_ind])
    alldata.append(tr_data)
    row_endings.append(tr_data[-int(2 * period):])

    # print(tr_data)

    length = len(tr_data)
    start = 0
    end = length-int(2*period)-horizon
    if end > 0:
        window_mp = 2
    else:
        number_of_windows_for_row.append(0)
        continue
    added = 0
    for i in range(start, end):
        on_row = tr_data[i:i+int(window_mp*period)]
        # if np.count_nonzero(on_row) == 0:
        #     continue
        windows.append(on_row)
        added += 1
    # print(added, start, end, length, horizon, 2*period)
    # input()
    number_of_windows_for_row.append(added)
data.close()
print("Items:", len(windows))
print("Lookup:", sum(number_of_windows_for_row))

n_of_endings_to_process = max(int(100000000 / len(windows)), 1)
# n_of_endings_to_process = max(int(50000000 / len(windows)), 1)
print("jump", n_of_endings_to_process)
counter = 0
for gap in range(0, len(alldata), n_of_endings_to_process):
    corr_matrix = corr2_coeff(np.array(row_endings[gap:gap + n_of_endings_to_process]), np.array(windows))
    print("Matrix done", time()-now)
    # print(corr_matrix.shape)

    # Trim transforms to same length for comparisons
    # tr_length = min([len(tf) for tf in row_transforms])
    # tr_length = period+1
    # row_transforms = [tf[:tr_length] for tf in row_transforms]
    
    # Start working on predicting
    # data = open(path_to)
    
    period = peris[timez]
    horizon = horis[timez]
    # row_index = 0
    for corr_matrix_index, all_data_index in enumerate(range(gap, min(gap + n_of_endings_to_process, len(alldata)))):
        data_row = alldata[all_data_index]
        # if all_data_index % 2000 == 0:
        #     print("at", all_data_index, time()-now)
        # if row_index < 40000 or rownames[row_ind2] in doneset:
        #     row_index += 1
        #     continue
        counter += 1
        name = rownames[all_data_index]
        # tr_data = np.array([float(num) for num in row[name_ind+1:] if num.strip() != ""])
        result = predict(data_row, horizon, period, number_of_windows_for_row, alldata, all_data_index, corr_matrix, name, 1, corr_matrix_index)
        f.write(name)
        for num in result:
            # f.write("," + str(num))
            if num > 0:
                f.write(","+str(num))
            else:
                f.write(",0")
        f.write("\n")
        # row_index += 1
    f.flush()
f.close()
