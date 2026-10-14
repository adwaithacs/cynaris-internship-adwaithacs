
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test data
df = pd.read_csv("healthcare_faq_50.csv")
X = df["text"]
y = df["intent"]

models = {
    "Original": "healthwire_week2_model.joblib",
    "Improved": "healthwire_week2_model_improved.joblib"
}

for name, filename in models.items():
    model = joblib.load(filename)
    predictions = model.predict(X)

    print(f"\n--- {name} Model ---")
    print(f"Accuracy: {accuracy_score(y, predictions):.2%}")
    print(classification_report(y, predictions, zero_division=0))

    results = df.copy()
    results["predicted_intent"] = predictions
    results["correct"] = results["intent"] == predictions
    results.to_csv(f"{name.lower()}_model_test_results.csv", index=False)

print("\nComparison complete. Original model remains unchanged.")
