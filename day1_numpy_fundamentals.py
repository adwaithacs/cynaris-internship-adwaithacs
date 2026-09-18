import numpy as np

# DAY 1 - NUMPY FUNDAMENTALS

# Load Iris CSV dataset
data = np.genfromtxt(
    "Iris.csv",
    delimiter=",",
    skip_header=1,
    usecols=(1, 2, 3, 4),
    dtype=float,
    invalid_raise=False
)

# Remove incomplete rows
data = data[~np.isnan(data).any(axis=1)]

# Display results
print("Iris Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data[:5])

print("\nMean of Each Feature:")
print(np.mean(data, axis=0))

print("\nMaximum of Each Feature:")
print(np.max(data, axis=0))
print("\nStandard Deviation of Each Feature:")
print(np.std(data, axis=0))

print("\nVariance of Each Feature:")
print(np.var(data, axis=0))
print("/nTotal number of data rows:")
print(len(data))