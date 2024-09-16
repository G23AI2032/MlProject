import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np

# Load the dataset
file_path = 'Stress.csv'
data = pd.read_csv(file_path)

# Prepare the data
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Convert sparse matrices to dense because neural networks work with dense inputs
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# 2. Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_dense, y_train)

# Convert labels to categorical (one-hot encoding)
y_train_balanced = to_categorical(y_train_balanced, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 3. Build the Neural Network Model
model = Sequential()

# Input layer (size of the TF-IDF features)
model.add(Dense(512, input_shape=(X_train_balanced.shape[1],), activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Hidden layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output layer (binary classification, so we use 2 units with softmax)
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train the model
history = model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

# 5. Evaluate the model
y_pred = model.predict(X_test_dense)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test_classes, y_pred_classes)
report = classification_report(y_test_classes, y_pred_classes)

print(f"Neural Network Accuracy: {accuracy}")
print("Neural Network Classification Report:\n", report)
