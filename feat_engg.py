import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier

# Load the dataset
file_path = 'Stress.csv'
data = pd.read_csv(file_path)

# Prepare the data
X = data['text']
y = data['label']

# 1. TF-IDF Vectorization with n-grams and stop word removal
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)  # Transform the entire dataset's text

# 2. One-hot encode metadata (e.g., 'subreddit') on the entire dataset
subreddit_ohe = OneHotEncoder(sparse_output=False)
X_subreddit = subreddit_ohe.fit_transform(data[['subreddit']])  # One-hot encode the entire subreddit column

# 3. Concatenate TF-IDF features with one-hot encoded 'subreddit'
X_full = hstack([X_tfidf, X_subreddit])

# 4. Split the full concatenated data (X_full) and labels (y) into train and test sets
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# 3. Handle class imbalance using SMOTE (optional, if imbalance exists)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_full, y_train)

# 4. Random Forest Classifier with hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy')
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test_full)

print(f"Random Forest Best Params: {rf_model.best_params_}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# 5. Support Vector Machine Classifier with default parameters
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train_full, y_train)
svm_pred = svm_model.predict(X_test_full)

print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred)}")
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# 6. XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_full, y_train)
xgb_pred = xgb_model.predict(X_test_full)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)}")
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# 7. Ensemble Voting Classifier (Random Forest + SVM + XGBoost)
voting_model = VotingClassifier(estimators=[
    ('rf', rf_model.best_estimator_),
    ('svc', SVC(probability=True, class_weight='balanced')),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
], voting='soft')

voting_model.fit(X_train_balanced, y_train_balanced)
voting_pred = voting_model.predict(X_test_full)

print(f"Voting Classifier Accuracy: {accuracy_score(y_test, voting_pred)}")
print("Voting Classifier Classification Report:\n", classification_report(y_test, voting_pred))
