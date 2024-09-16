from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = pd.read_csv('Stress.csv')

# Prepare the data
X = data['text']
y = data['label']
# We already have X_train_tfidf and X_test_tfidf from the previous step
# You can uncomment the below lines if X_train_tfidf and X_test_tfidf are not ready
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Random Forest Classification Report:\n", rf_report)

# Support Vector Machine Classifier
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred)

print(f"SVM Accuracy: {svm_accuracy}")
print("SVM Classification Report:\n", svm_report)
