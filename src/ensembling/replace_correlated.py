
import pandas as pd
import numpy as np
import os
import time


correlations_file_path = os.path.join(os.pardir, os.pardir, "correlations", "correlator", "correlator_correlations_inf.csv")
forecasts_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "correlator", "correlator_forecasts_inf.csv")
ensemble_forecasts_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "ensemble", "ensemble.csv")

correlations_file = pd.read_csv(correlations_file_path, header=None)
forecasts_file = pd.read_csv(forecasts_file_path)
ensemble_forecasts_file = pd.read_csv(ensemble_forecasts_file_path)

threshold = 0.9999
str_threshold = str(threshold).replace(".", "")
resulting_forecasts_file_path = os.path.join(os.pardir, os.pardir, "forecasts", "ensemble", "ensemble_with_" + os.path.basename(forecasts_file_path)[:-4] + "_" + str_threshold + "_" + str(time.time()) + ".csv")

correlations_file = correlations_file.loc[correlations_file.iloc[:, 0].str.startswith('D')]
forecasts_file = forecasts_file.loc[forecasts_file.iloc[:, 0].str.startswith('D')]

correlations_file.set_index(keys=correlations_file.columns[0], inplace=True)
forecasts_file.set_index(keys=forecasts_file.columns[0], inplace=True)
ensemble_forecasts_file.set_index(keys=ensemble_forecasts_file.columns[0], inplace=True)

correlations_dict = correlations_file.to_dict()[1]

replaced = []
final_result = pd.DataFrame(index=["D"+str(i) for i in range(1, 4228)], columns=[str(i) for i in range(0, 14)])
for j, (idx, row) in enumerate(ensemble_forecasts_file.iterrows()):
    final_result.loc[idx] = pd.DataFrame(ensemble_forecasts_file.loc[idx, :]).transpose().loc[idx]
    final_result.append(pd.DataFrame(ensemble_forecasts_file.loc[idx, :]).transpose())
    if idx in correlations_dict:
        correlation = float(correlations_dict[idx])
        if correlation > threshold and not np.isnan(correlation):
            replaced.append(idx)
            final_result.loc[idx] = pd.DataFrame(forecasts_file.loc[idx, :]).transpose().loc[idx]

print(replaced)
print(len(replaced))

print(final_result)
pd.DataFrame(final_result).to_csv(resulting_forecasts_file_path)
