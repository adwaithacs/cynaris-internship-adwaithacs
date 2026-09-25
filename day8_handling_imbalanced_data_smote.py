import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Loading dataset
df = pd.read_csv("day8_dataset.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check the target distribution
print("\nOriginal stroke distribution:")
print(df["stroke"].value_counts())

# Handle missing BMI values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Remove ID column
df = df.drop("id", axis=1)

# Encode categorical columns
categorical_columns = ["gender","ever_married","work_type","Residence_type","smoking_status"]
encoder = LabelEncoder()
for column in categorical_columns:df[column] = encoder.fit_transform(df[column])

# Separate features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

# Display the new class distribution
print("\nDistribution after SMOTE:")
print(y_train_smote.value_counts())