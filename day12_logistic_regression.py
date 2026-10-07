import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("day12_dataset.csv")
print(df.head())
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Preprocess
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_transformer = Pipeline([
    ("fill", __import__("sklearn").impute.SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_transformer = Pipeline([
    ("fill", __import__("sklearn").impute.SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Did Not Survive", "Survived"]
))

# Confusion matrix plot
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
).plot()
plt.title("Confusion Matrix")
plt.show()

# Feature coefficients
features_out = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "Feature": features_out,
    "Coefficient": coefficients
})

coef_df["Absolute Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(
    "Absolute Coefficient",
    ascending=False
)

print("\nTop 10 Influential Features:")
print(coef_df.head(10))

# C comparison
print("\n========== C COMPARISON ==========")

for c in [0.01, 0.1, 1, 10, 100]:
    c_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(C=c, max_iter=1000, random_state=42))
    ])

    c_model.fit(X_train, y_train)
    c_pred = c_model.predict(X_test)
    c_prob = c_model.predict_proba(X_test)[:, 1]

    print(
        f"C={c:<5} "
        f"Accuracy={accuracy_score(y_test, c_pred):.4f} "
        f"ROC-AUC={roc_auc_score(y_test, c_prob):.4f}"
    )
