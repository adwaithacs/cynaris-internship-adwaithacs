import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Load the COVID-19 dataset
df = pd.read_csv("day9_dataset.csv")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Select numerical features
features = ["Total Cases","Active","Discharged","Deaths","Active Ratio","Discharge Ratio","Population"]
X = df[features]
y = df["Death Ratio"]
print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Standard Scaling
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
print("\nStandardScaler:")
print("Training data (first 5 rows):")
print(X_train_standard[:5])
print("\nTesting data (first 5 rows):")
print(X_test_standard[:5])

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)
print("\nMinMaxScaler:")
print("Training data (first 5 rows):")
print(X_train_minmax[:5])
print("\nTesting data (first 5 rows):")
print(X_test_minmax[:5])

#Robust Scaling
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)
print("\nRobustScaler:")
print("Training data (first 5 rows):")
print(X_train_robust[:5])
print("\nTesting data (first 5 rows):")
print(X_test_robust[:5])

# Cross-validation
model = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model,X,y,cv=kfold,scoring="r2")
print("\nCross-Validation R² Scores:")
print(cv_scores)
print("\nMean Cross-Validation R² Score:")
print(cv_scores.mean())

# Summary
print("\n--- W2D4 Summary ---")
print("Train/Test Split: 80% training, 20% testing")
print("Scaling methods: StandardScaler, MinMaxScaler, RobustScaler")
print("Cross-Validation: 5-Fold")
print("W2D4 practical task completed successfully.")

