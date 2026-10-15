
import json
import pandas as pd

symptoms = pd.read_csv("aiml_healthcare_symptoms.csv")

with open("aiml_labelled_samples.json", encoding="utf-8") as file:
    samples = json.load(file)

faq = pd.DataFrame(samples)

print("Symptoms Dataset:", symptoms.shape)
print(symptoms.head())

print("\nFAQ Dataset:", faq.shape)
print(faq.head())

print("\nIntent Counts:")
print(faq["intent"].value_counts())


print("\n--- Symptoms Data Quality ---")
print("Missing values:")
print(symptoms.isnull().sum())
print("Duplicate rows:", symptoms.duplicated().sum())

print("\n--- FAQ Data Quality ---")
print("Missing values:")
print(faq.isnull().sum())
print("Duplicate rows:", faq.duplicated().sum())
print("Duplicate questions:", faq["text"].duplicated().sum())

print("\n--- FAQ Confidence Statistics ---")
print(faq["confidence"].describe())


# Clean FAQ data
faq["text"] = faq["text"].str.strip().str.lower()
faq["intent"] = faq["intent"].str.strip().str.lower()
faq = faq.dropna(subset=["text", "intent"])
faq = faq.drop_duplicates(subset=["text", "intent"])
faq = faq.reset_index(drop=True)

print("\n--- Cleaned FAQ Dataset ---")
print("Rows after cleaning:", len(faq))
print("Missing values:\n", faq.isnull().sum())
print("Duplicate questions:", faq["text"].duplicated().sum())
print("Intent counts:\n", faq["intent"].value_counts())

faq.to_csv("cleaned_faq_dataset.csv", index=False)
print("Cleaned dataset saved successfully!")


# Add FAQ examples
extra_faq = [
    ("my temperature is 101 and i feel weak", "fever"),
    ("i have a high fever and headache", "fever"),
    ("feeling feverish since yesterday", "fever"),
    ("why do i get chills with fever", "fever"),
    ("my body feels hot and tired", "fever"),

    ("could fever and chills be malaria", "malaria"),
    ("i have recurring fever and sweating", "malaria"),
    ("what are common signs of malaria", "malaria"),
    ("can mosquito bites spread malaria", "malaria"),
    ("i have chills and headache after travel", "malaria"),

    ("i feel thirsty all the time", "diabetes"),
    ("does frequent urination indicate diabetes", "diabetes"),
    ("what are early signs of diabetes", "diabetes"),
    ("my blood sugar reading is high", "diabetes"),
    ("can diabetes cause tiredness", "diabetes"),

    ("i have a runny nose and sneezing", "cold_flu"),
    ("my throat hurts and i am coughing", "cold_flu"),
    ("i have congestion and a mild cough", "cold_flu"),
    ("how can i tell if i have a cold", "cold_flu"),
    ("i feel achy with a blocked nose", "cold_flu"),

    ("my blood pressure reading is high", "hypertension"),
    ("what does high blood pressure mean", "hypertension"),
    ("can hypertension have no symptoms", "hypertension"),
    ("how is high blood pressure monitored", "hypertension"),
    ("i need information about blood pressure", "hypertension"),

    ("what can this chatbot help me with", "general_query"),
    ("can you give general health information", "general_query"),
    ("where can i get medical advice", "general_query"),
    ("i have a question about my health", "general_query"),
    ("what topics can i ask about", "general_query"),

    
    ("i feel thirsty all the time", "diabetes"),
    ("what is insulin dosage", "diabetes"),
    ("how is insulin used in diabetes", "diabetes"),
    ("how can i tell if i have a cold", "cold_flu"),
    ("what are common cold symptoms", "cold_flu"),
    ("can hypertension have no symptoms", "hypertension"),
    ("does high blood pressure always cause symptoms", "hypertension"),

]

extra_df = pd.DataFrame(extra_faq, columns=["text", "intent"])
faq = pd.concat(
    [faq[["text", "intent"]], extra_df],
    ignore_index=True
)
faq = faq.drop_duplicates(subset=["text", "intent"])
print("Expanded FAQ rows:", len(faq))
print(faq["intent"].value_counts())




# Week 2 model evaluation
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

X = faq["text"]
y = faq["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("classifier", LinearSVC(class_weight="balanced"))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n--- Week 2 Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

labels = sorted(y.unique())
matrix = confusion_matrix(y_test, predictions, labels=labels)

plt.figure(figsize=(9, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("HealthWire Confusion Matrix")
plt.tight_layout()
plt.savefig("week2_confusion_matrix.png")
plt.show()

joblib.dump(model, "healthwire_week2_model.joblib")
print("\nModel saved: healthwire_week2_model.joblib")
print("Confusion matrix saved: week2_confusion_matrix.png")

predictions = model.predict(X_test)


# Check mistakes
check = pd.DataFrame({
    "Question": X_test.values,
    "Actual": y_test.values,
    "Predicted": predictions
})

wrong = check[check["Actual"] != check["Predicted"]]

print("\n--- Incorrect Predictions ---")
print(wrong.to_string(index=False))

