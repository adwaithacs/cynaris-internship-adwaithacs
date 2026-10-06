import pandas as pd

# Load Titanic dataset
df = pd.read_csv("day10_dataset.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin because it has too many missing values
df = df.drop(columns=["Cabin"])

# Check missing values again
print("\nMissing Values After Handling:")
print(df.isnull().sum())

from sklearn.preprocessing import OneHotEncoder

# Convert categorical columns into numbers
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_data = encoder.fit_transform(df[["Sex", "Embarked"]])

# Convert encoded data into a DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(["Sex", "Embarked"]),
    index=df.index
)

# Remove original categorical columns
df = df.drop(columns=["Sex", "Embarked"])

# Add encoded columns
df = pd.concat([df, encoded_df], axis=1)

print("\nDataset After Encoding:")
print(df.head())

print("\nNew Columns:")
print(df.columns.tolist())

from sklearn.preprocessing import StandardScaler

# Select numerical columns to scale
numeric_columns = ["Age", "SibSp", "Parch", "Fare"]

# Create scaler
scaler = StandardScaler()

# Apply scaling
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nDataset After Scaling:")
print(df[numeric_columns].head())

print("\nMean after scaling:")
print(df[numeric_columns].mean())

print("\nStandard deviation after scaling:")
print(df[numeric_columns].std())

# Remove text columns that are not used for ML
df = df.drop(columns=["Name", "Ticket"])

# Save the processed dataset
df.to_csv("day10_ml_ready_titanic.csv", index=False)

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Dataset Columns:")
print(df.columns.tolist())

print("\nML-ready dataset saved successfully!")