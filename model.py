# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import re

# Load the dataset
real = pd.read_csv("Dataset/train/True.csv")
fake = pd.read_csv("Dataset/train/Fake.csv")

real["label"] = 0
fake["label"] = 1

df = pd.concat([real, fake], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

## view data
print(df.head())
print(df["label"].value_counts())

# Preprocessing
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    return text

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
try:
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    print("Model and vectorizer saved successfully.")

except Exception as e:
    print("Error saving model and vectorizer:", e)
