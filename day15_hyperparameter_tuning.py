
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
df = pd.read_csv("Iris.csv")

# 2. Select features and target
X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]
y = df["Species"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define the SVM model and parameters
svm = SVC()
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

# 6. Perform Grid Search
grid_search = GridSearchCV(
    svm, param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

print("Best GridSearch Parameters:", grid_search.best_params_)
print("Best GridSearch CV Accuracy:",
      round(grid_search.best_score_ * 100, 2), "%")

grid_predictions = grid_search.predict(X_test)
print("GridSearch Test Accuracy:",
      round(accuracy_score(y_test, grid_predictions) * 100, 2), "%")

# 7. Perform Randomized Search
random_search = RandomizedSearchCV(
    svm, param_grid, n_iter=8, cv=5,
    scoring="accuracy", random_state=42
)
random_search.fit(X_train, y_train)

print("\nBest RandomizedSearch Parameters:",
      random_search.best_params_)
print("Best RandomizedSearch CV Accuracy:",
      round(random_search.best_score_ * 100, 2), "%")

random_predictions = random_search.predict(X_test)
print("RandomizedSearch Test Accuracy:",
      round(accuracy_score(y_test, random_predictions) * 100, 2), "%")

# Compare GridSearchCV and RandomizedSearchCV results
print("\n--- Hyperparameter Tuning Comparison ---")

print(f"GridSearchCV best parameters: {grid_search.best_params_}")
print(f"RandomizedSearchCV best parameters: {random_search.best_params_}")

print(f"GridSearchCV CV accuracy: {grid_search.best_score_ * 100:.2f}%")
print(f"RandomizedSearchCV CV accuracy: {random_search.best_score_ * 100:.2f}%")

if grid_search.best_score_ > random_search.best_score_:
    print("GridSearchCV performed better.")
elif random_search.best_score_ > grid_search.best_score_:
    print("RandomizedSearchCV performed better.")
else:
    print("Both methods achieved the same CV accuracy.")
