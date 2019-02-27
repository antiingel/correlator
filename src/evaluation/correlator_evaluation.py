# Base modules
import os

# data structure modules
from collections import defaultdict
from itertools import chain
import numpy as np
import pandas as pd

# data modules
from src.evaluation.load_data import load_all_data_pd
from src.evaluation.scoring_functions import sMAPE, MASE, OWA


threshold = 0.99
str_threshold = str(threshold).replace(".", "")

input_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "correlator", "correlator_forecasts_inf.csv")
path_naive = os.path.join(os.pardir, os.pardir, "forecasts", "single_methods", "naive.csv")
correlations_file = os.path.join(os.pardir, os.pardir, "correlations", "correlator", "correlator_correlations_inf.csv")
score_file_path = os.path.join(os.pardir, os.pardir, "scores", "correlator", os.path.basename(input_file_path)[:-4] + "_" + str_threshold + ".csv")

correlations_file_lines = open(correlations_file, "r").readlines()

print(input_file_path)

season_horizon = {"Daily": (1, 14)}

path_train = os.path.join(os.pardir, os.pardir, "data", "full", "train")
path_test = os.path.join(os.pardir, os.pardir, "data", "full", "test")


meta = pd.read_csv(os.path.join(os.pardir, os.pardir, "data", "M4Info.csv"))
# Training data files ... just load them all in ....
files = ["Daily.csv"]
raw_data_filler_cols = (1, 0)


category = "Daily"


# Change paths for input (or make these parameters for the script...)

def huge_damn_scorer(input_file_path, eval_categories=[category]):
    (train_da, test_da), = load_all_data_pd(path_train, path_test, files)
    all_data = {category: (train_da, test_da)}
    test_res = pd.read_csv(input_file_path)
    test_naive = pd.read_csv(path_naive)

    naive_mases = defaultdict(lambda: defaultdict(list))
    naive_smapes = defaultdict(lambda: defaultdict(list))
    mases = defaultdict(lambda: defaultdict(list))
    smapes = defaultdict(lambda: defaultdict(list))

    for freq in eval_categories:
        subset = meta.loc[meta.SP == freq]
        train, test = all_data[freq]
        season, horizon = season_horizon[freq]

        just_a_check = subset.shape[0]
        checker_juster = 0

        correlator_overall_counter = 0
        correlator_counter_list = []
        category_list = []
        for cat in subset.category.unique():
            print(freq, cat)
            category_list.append(cat)
            m4ids = subset.loc[subset.category == cat].M4id
            sub_train = train.loc[train["V1"].isin(m4ids)]
            sub_res = test_res.loc[test_res["V1"].isin(m4ids)].dropna(axis=1)
            sub_naive = test_naive.loc[test_naive["V1"].isin(m4ids)].dropna(axis=1)

            # print(sub_train["V1"].tolist())
            # print(sub_res["V1"].tolist())
            # print(sub_naive["V1"].tolist())
            for a, b in zip(sub_train["V1"].tolist(), sub_res["V1"].tolist()):
                if a != b:
                    print(a, b)
            assert sub_train["V1"].tolist() == sub_res["V1"].tolist()
            assert sub_train["V1"].tolist() == sub_naive["V1"].tolist()

            correlator_counter = 0
            for i, (subtrain_idx, subtrain_row) in enumerate(sub_train.iterrows()):
                # print(subtrain_row)
                correlations_line_split = correlations_file_lines[int(subtrain_row["V1"][1:])-1].strip().split(",")
                # print(correlations_line_split, subtrain_idx)
                correlation_index = correlations_line_split[0].strip('"')
                correlation = float(correlations_line_split[1])
                assert correlation_index == subtrain_row["V1"]
                if correlation > threshold and not np.isnan(correlation):
                    # print(correlation)
                    correlator_counter += 1

                    test_row = test.iloc[subtrain_idx]
                    train_row = np.array(subtrain_row[raw_data_filler_cols[0]:].dropna().values, dtype=np.float32)
                    test_real = np.array(test_row.dropna()[raw_data_filler_cols[1]:])

                    result_row = np.array(sub_res.iloc[i, 1:], dtype=np.float32).flatten()
                    naive_row = np.array(sub_naive.iloc[i, 1:], dtype=np.float32).flatten()

                    tmp1 = MASE(train_row, test_real, naive_row, season)
                    tmp2 = sMAPE(test_real, naive_row)

                    if not np.isinf(tmp1):
                        naive_mases[freq][cat].append(tmp1)
                    else:
                        print("naive mase nan")
                    if not np.isinf(tmp2):
                        naive_smapes[freq][cat].append(tmp2)
                    else:
                        print("naive smape nan")
                    tmp1 = MASE(train_row, test_real, result_row, season)
                    tmp2 = sMAPE(test_real, result_row)
                    if not np.isinf(tmp1):
                        mases[freq][cat].append(tmp1)
                    else:
                        print("mase nan")
                    if not np.isinf(tmp2):
                        smapes[freq][cat].append(tmp2)
                    else:
                        print("smape nan")

            print(correlator_counter)
            correlator_counter_list.append(correlator_counter)
            correlator_overall_counter += correlator_counter
            checker_juster += len(list(m4ids))
        assert checker_juster == just_a_check
        print(correlator_overall_counter)
        print("\t".join(category_list))
        print("\t".join(map(str, correlator_counter_list)))
    print("Starting evaluating")
    df = pd.DataFrame(index=list(mases.keys()) + ["All"], columns=(list(mases[eval_categories[0]]) + ["All"]))

    overall_naive_mases = []
    overall_naive_smapes = []
    overall_mases = []
    overall_smapes = []

    levels_mases = defaultdict(list)
    levels_naive_mases = defaultdict(list)
    levels_smapes = defaultdict(list)
    levels_naive_smapes = defaultdict(list)

    for cat in mases:
        all_cat_naive_mases = []
        all_cat_naive_smapes = []
        all_cat_mases = []
        all_cat_smapes = []
        for h in mases[cat]:
            cur_naive_mases = naive_mases[cat][h]
            cur_naive_smapes = naive_smapes[cat][h]
            cur_mases = mases[cat][h]
            cur_smapes = smapes[cat][h]

            cur_mases_mean = np.mean(cur_mases)
            cur_smapes_mean = np.mean(cur_smapes)
            df.loc[cat + " MASE", h] = cur_mases_mean
            df.loc[cat + " SMAPE", h] = cur_smapes_mean
            df.loc[cat + " OWA", h] = OWA(cur_mases_mean, cur_smapes_mean, np.mean(cur_naive_mases), np.mean(cur_naive_smapes))

            levels_mases[h].append(cur_mases)
            levels_smapes[h].append(cur_smapes)
            levels_naive_mases[h].append(cur_naive_mases)
            levels_naive_smapes[h].append(cur_naive_smapes)

            all_cat_naive_mases.append(cur_naive_mases)
            all_cat_naive_smapes.append(cur_naive_smapes)
            all_cat_mases.append(cur_mases)
            all_cat_smapes.append(cur_smapes)

        naive_mase_mean = np.mean(list(chain(*all_cat_naive_mases)))
        naive_smape_mean = np.mean(list(chain(*all_cat_naive_smapes)))
        score_mase = np.mean(list(chain(*all_cat_mases)))
        score_smape = np.mean(list(chain(*all_cat_smapes)))

        overall_naive_mases.append(all_cat_naive_mases)
        overall_naive_smapes.append(all_cat_naive_smapes)
        overall_mases.append(all_cat_mases)
        overall_smapes.append(all_cat_smapes)
        df.loc[cat + " MASE", "All"] = score_mase
        df.loc[cat + " SMAPE", "All"] = score_smape
        df.loc[cat + " OWA", "All"] = OWA(score_mase, score_smape, naive_mase=naive_mase_mean, naive_smape=naive_smape_mean)

    for h in levels_mases.keys():
        naive_mase_mean = np.mean(list(chain(*levels_naive_mases[h])))
        naive_smape_mean = np.mean(list(chain(*levels_naive_smapes[h])))
        score_mase = np.mean(list(chain(*levels_mases[h])))
        score_smape = np.mean(list(chain(*levels_smapes[h])))
        df.loc["All" + " MASE", h] = score_mase
        df.loc["All" + " SMAPE", h] = score_smape
        df.loc["All" + " OWA", h] = OWA(score_mase, score_smape, naive_mase=naive_mase_mean, naive_smape=naive_smape_mean)

    overall_naive_mase = np.mean(list(chain(*list(chain(*overall_naive_mases)))))
    overall_naive_smape = np.mean(list(chain(*list(chain(*overall_naive_smapes)))))
    overall_mase = np.mean(list(chain(*list(chain(*overall_mases)))))
    overall_smape = np.mean(list(chain(*list(chain(*overall_smapes)))))
    # print(overall_naive_mase, overall_naive_smape)
    df.loc["All" + " MASE", "All"] = overall_mase
    df.loc["All" + " SMAPE", "All"] = overall_smape
    df.loc["All" + " OWA", "All"] = OWA(overall_mase, overall_smape, overall_naive_mase, overall_naive_smape)
    return df


result = huge_damn_scorer(input_file_path)
print(result)
result.to_csv(score_file_path, sep="\t")
