
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
df = pd.read_csv("Iris.csv")

# Prepare features and target
X = df[["SepalLengthCm", "SepalWidthCm",
        "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features (important for SVM and KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Compare model accuracy
print("SVM Accuracy:",
      round(accuracy_score(y_test, svm_predictions) * 100, 2), "%")

print("KNN Accuracy:",
      round(accuracy_score(y_test, knn_predictions) * 100, 2), "%")

# Display classification report
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# Compare different K values for KNN
print("\nKNN Accuracy for Different K Values:")

for k in [3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"K = {k}: {accuracy * 100:.2f}%")