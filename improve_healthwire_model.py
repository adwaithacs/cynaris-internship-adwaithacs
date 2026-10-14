
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load data
with open("aiml_labelled_samples.json", encoding="utf-8") as file:
    samples = json.load(file)

faq = pd.DataFrame(samples)
faq["text"] = faq["text"].str.strip().str.lower()
faq["intent"] = faq["intent"].str.strip().str.lower()
faq = faq.dropna(subset=["text", "intent"])
faq = faq.drop_duplicates(subset=["text", "intent"])

# Add targeted examples
extra_faq = [
    ("my temperature is high and i feel feverish", "fever"),
    ("i feel hot and feverish tonight", "fever"),
    ("how should i monitor a fever at home", "fever"),
    ("how do i check my temperature when i have a fever", "fever"),

    ("my blood glucose level is high", "diabetes"),
    ("what does a high blood sugar result mean", "diabetes"),
    ("please explain hba1c", "diabetes"),
    ("what is hba1c used for", "diabetes"),
    ("i need help managing diabetes", "diabetes"),
    ("where can i get diabetes advice", "diabetes"),
    ("i am thirsty and urinating frequently", "diabetes"),
    ("i feel unusually thirsty and need to urinate often", "diabetes"),
    ("how can i check my blood sugar at home", "diabetes"),
    ("my feet feel numb and i have diabetes", "diabetes"),

    ("should i get a flu vaccine", "cold_flu"),
    ("where can i ask about flu vaccination", "cold_flu"),
    ("how can i prevent spreading a cold at home", "cold_flu"),
    ("how do i avoid spreading flu to my family", "cold_flu"),

    ("i want advice about high blood pressure", "hypertension"),
    ("what foods are suitable for high blood pressure", "hypertension"),

    ("how do i renew my prescription", "general_query"),
    ("where can i get prescription renewal help", "general_query"),
    ("how do i book a lab test", "general_query"),
    ("where can i ask about clinic services", "general_query"),
    ("how do doctors test for malaria", "malaria"),
    ("where can i ask about a health insurance reimbursement", "general_query"),
    ("where do i ask about my medical records", "general_query"),
]


extra_df = pd.DataFrame(extra_faq, columns=["text", "intent"])
faq = pd.concat([faq[["text", "intent"]], extra_df], ignore_index=True)
faq = faq.drop_duplicates(subset=["text", "intent"])

# Train model
X_train, X_val, y_train, y_val = train_test_split(
    faq["text"],
    faq["intent"],
    test_size=0.25,
    random_state=42,
    stratify=faq["intent"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("classifier", LinearSVC(class_weight="balanced"))
])

model.fit(X_train, y_train)
predictions = model.predict(X_val)

print("Validation accuracy:", round(accuracy_score(y_val, predictions), 4))
print(classification_report(y_val, predictions, zero_division=0))

# Save separately
joblib.dump(model, "healthwire_week2_model_improved.joblib")
print("Saved: healthwire_week2_model_improved.joblib")
