
import os

unfiltered_correlations = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "unfiltered_correlations.csv")).readlines()
filter = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "filter.csv")).readlines()
filtered_correlations = open(os.path.join(os.pardir, os.pardir, "correlations", "analysis", "filtered_correlations.csv"), "w")


filtered = 0
not_filtered = 0
for unfiltered_correlation in unfiltered_correlations:
    if unfiltered_correlation in filter:
        filtered_correlations.write(unfiltered_correlation)
        not_filtered += 1
    else:
        filtered += 1

filtered_correlations.close()

print(filtered, not_filtered, len(filter), len(unfiltered_correlations))
