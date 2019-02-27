# Correlator for finding correlated time series
## Getting Started

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/open?id=1o7xC4j2ai01st8dL_KUcM6gB_ThPpPrF). The original dataset can also be found [here](https://www.mcompetitions.unic.ac.cy/the-dataset/), but using the previous link is recommended, because it contains folder structure required by the code. Extract content of `correlator.zip` to the same folder that contains `src`.

### Content of `correlator.zip`

The `data` folder contains the original dataset (in folder `full`) and our internal train-test split (in folder `holdout`). There is also `M4info.csv` file which contains information about the time series.

The `forecasts` folder contains the forecasts that our code produces with different settings. In more detail:
 * folder `single_methods` contains forecasts for 5 models (2 ARIMA models, ETS, Naive and a custom method),
 * folder `ensemble` contains forecasts of our ensemble of the 5 single methods and also forecasts of a combination of ensemble and correlator,
 * folder `correlator` contains forecasts of the correlator.
 
The content of `scores` folder is organised similarly to `forecasts` folder and it contains MASE, sMAPE and OWA performance measures (see [here](https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf) for details) for the corresponding forecasts in `forecasts` folder. In addition, it contains a folder `analysis` which has MASE, sMAPE and OWA for different similarity categories that our analysis revealed (see our article for more details).

Finally, the folder `correlations` has two subfolders. The first subfolder `analysis` contains information about "global" correlations that our method found and the second subfolder `correlator` contains information about "local" correlations that our method found.

### Content of `src`

1. Folder `single_methods` contains code for all of the single method based predictors except Naive. They reads data from `data/full/train` and produces forecasts into `forecasts/single_methods` folder. Custom method also uses the holdout data.
2. Folder `ensemble` contains the code of ensemble. File `ensemble.py` reads input data from `forecasts/single_methods` and produces forecasts into `forecasts/ensemble` folder. File `replace_correlated.py` can be used after running the correlator to replace some of the ensemble forecasts with correlator forecasts. It reads correlation information from `correlations/correlator` and forecasts from `forecasts/ensemble` and `forecasts/correlator`.
3. Folder `correlator` contains code of the correlator. It reads data from `data/full/train` and writes found correlations into `correlations/correlator` and corresponding forecasts into `forecasts/correlator`.
4. Folder `correlations_analysis` contains code used to find "global" correlations. File `cross_correlation.py` reads data from `data/full/train` and writes found correlations into `correlations/analysis`. File `filter_correlations.py` simulates manual filtering we did to the correlations found by `cross_correlation.py`. It reads the output of `cross_correlation.py` from `correlations/analysis` and information about which correlations to keep from `correlations/analysis/filter.csv`.
5. Folder `evaluation` contains code that is used to evaluate the results. It reads forecasts from the corresponding folder in `forecasts` and produces scores to corresponding folder in `scores`. In more detail, `evaluation.py` can be used to evaluate single methods and ensemble (also ensemble with correlator). File `correlator_evaluation.py` can be used to evaluate correlator results (that is only the time series for which correlator made a prediction are considered). Finally, `correlation_analysis_evaluation.py` can be used to find MASE, sMAPE and OWA for different similarity categories that our analysis revealed.

## Contact

For additional information, feel free to contact Anti Ingel (antiingel@gmail.com).
