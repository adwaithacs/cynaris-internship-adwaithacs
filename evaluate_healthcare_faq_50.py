
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from pathlib import Path

# Load files
model = joblib.load("healthwire_week2_model.joblib")
df = pd.read_csv("healthcare_faq_50.csv")

# Validate data
if len(df) != 50:
    raise ValueError(f"Expected 50 questions, found {len(df)}")

if df[["text", "intent"]].isnull().any().any():
    raise ValueError("Missing text or intent values")

# Check training overlap
train_path = Path("cleaned_faq_dataset.csv")

if train_path.exists():
    train = pd.read_csv(train_path)
    train_texts = set(
        train["text"].astype(str).str.strip().str.lower()
    )
    test_texts = df["text"].astype(str).str.strip().str.lower()
    overlap = test_texts.isin(train_texts).sum()
    print("Exact training overlaps:", overlap)

# Predict
X = df["text"]
y = df["intent"]
predictions = model.predict(X)

# Metrics
print("\nHEALTHWIRE TEST RESULTS")
print("Questions:", len(df))
print(f"Accuracy: {accuracy_score(y, predictions):.2%}")
print(f"Precision: {precision_score(y, predictions, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y, predictions, average='weighted', zero_division=0):.4f}")
print(f"F1-score: {f1_score(y, predictions, average='weighted', zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y, predictions, zero_division=0))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y, predictions, xticks_rotation=45
)
plt.tight_layout()
plt.savefig("healthwire_50_question_confusion_matrix.png")
plt.show()

# Save predictions
df["predicted_intent"] = predictions
df["correct"] = df["intent"] == predictions
df.to_csv("healthwire_50_question_results.csv", index=False)

print("\nCorrect answers:", df["correct"].sum(), "/ 50")

print("\nIncorrect predictions:")
print(
    df.loc[
        ~df["correct"],
        ["text", "intent", "predicted_intent"]
    ].to_string(index=False)
)

print("\nResults saved successfully.")