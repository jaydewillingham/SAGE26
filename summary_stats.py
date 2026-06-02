import numpy as np
from scipy import stats

# Load data from text file
data = np.loadtxt("tdyngood_end_BulgeMass.txt")

# Summary statistics
summary = {
    "Count": len(data),
    "Mean": np.mean(data),
    "Median": np.median(data),
    "Standard Deviation": np.std(data, ddof=1),  # sample std
    "Variance": np.var(data, ddof=1),            # sample variance
    "Minimum": np.min(data),
    "Maximum": np.max(data),
    "Range": np.max(data) - np.min(data),
    "25th Percentile": np.percentile(data, 25),
    "75th Percentile": np.percentile(data, 75),
    "Interquartile Range": np.percentile(data, 75) - np.percentile(data, 25),
    "Skewness": stats.skew(data),
    "Kurtosis": stats.kurtosis(data)
}

# Print results
for key, value in summary.items():
    print(f"{key}: {value}")