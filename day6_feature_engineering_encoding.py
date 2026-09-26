import pandas as pd
import numpy as np
df = pd.read_csv("day6_dataset.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns
print("\nCategorical columns:")
print(categorical_cols)
print("\nNumerical columns:")
print(numerical_cols)

from sklearn.preprocessing import LabelEncoder

# Label Encoding
label_encoder = LabelEncoder()
df["Sex_Label"] = label_encoder.fit_transform(df["Sex"])
print("\nLabel Encoded Sex:")
print(df[["Sex", "Sex_Label"]].head())

#OneHot Encoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
result = encoder.fit_transform(df[["Embarked"]])
print("\nOne-Hot Encoded Embarked:")
print(result[:5])

#ordinal encoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
result = ordinal_encoder.fit_transform(df[["Pclass"]])
print("\nOrdinal Encoded Pclass:")
print(result[:5])

#Standardscalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
age_scaled = scaler.fit_transform(df[["Age"]])
print("\nStandard Scaled Age:")
print(age_scaled[:5])

#minmax scalar
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
age_minmax = minmax_scaler.fit_transform(df[["Age"]])
print("\nMinMax Scaled Age:")
print(age_minmax[:5])

#robustscalar
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
age_robust = robust_scaler.fit_transform(df[["Age"]])
print("\nRobust Scaled Age:")
print(age_robust[:5])

#plotDistribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(df["Age"].dropna(), bins=20, alpha=0.5, label="Original Age")
plt.hist(age_scaled, bins=20, alpha=0.5, label="StandardScaler")
plt.hist(age_minmax, bins=20, alpha=0.5, label="MinMaxScaler")
plt.hist(age_robust, bins=20, alpha=0.5, label="RobustScaler")
plt.title("Age Distribution Before and After Scaling")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

# Select numerical features
X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()

# Target variable
y = df["Survived"]

# Handle missing Age values
X["Age"] = X["Age"].fillna(X["Age"].median())

# Select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Display feature scores
feature_scores = pd.DataFrame({"Feature": X.columns,"Score": selector.scores_})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)
print("\nFeature Scores:")
print(feature_scores)
print("\nTop 5 Features:")
print(feature_scores.head(5))

# Feature Engineering Summary
# Encoding: LabelEncoder, OneHotEncoder, OrdinalEncoder
# Scaling: StandardScaler, MinMaxScaler, RobustScaler
# Feature Selection: SelectKBest