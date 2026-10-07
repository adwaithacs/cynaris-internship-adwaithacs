import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv("day11_dataset.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())

# Features and target
X = df.drop(columns=["Score"])
y = df["Score"]
print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Preprocess
categorical = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough")
print("Categorical columns:", categorical.tolist())

# Train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:")
print(y_pred)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))
print("R²:", round(r2, 4))

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[name] = [mse, rmse, mae, r2]

# Comparison
results_df = pd.DataFrame(
    results,
    index=["MSE", "RMSE", "MAE", "R²"]
).T

print("\nModel Comparison:")
print(results_df)

# Coefficients
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

features = linear_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = linear_model.named_steps["model"].coef_

print("\nCoefficients:")
print(pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}))

# Predicted vs actual
y_pred = linear_model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.show()

# Residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Self Review

print("\n========== SELF REVIEW ==========")
print("Linear Regression trained successfully.")
print("MSE, RMSE, MAE and R² calculated.")
print("Ridge and Lasso compared.")
print("Predicted vs Actual plot generated.")
print("Residual plot generated.")
print("Linear Regression coefficients displayed.")