
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Iris.csv")

# Prepare features and target
X = df[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]
y = df["Species"]

print("Dataset Shape:", df.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Compare accuracy
print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

# Feature importance
print("\nFeature Importance:")
print(pd.Series(rf.feature_importances_, index=X.columns))

print("\nProgram completed successfully!")