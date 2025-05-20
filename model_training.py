import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Step 1: Load data
df = pd.read_csv("FAQ Database.csv")

# Step 2: Clean
df = df.dropna(subset=["question", "intent"])
df["question"] = df["question"].str.strip().str.lower()
df["intent"] = df["intent"].str.strip().str.lower()

# Step 3: Vectorization + Model
X_train, X_test, y_train, y_test = train_test_split(df["question"], df["intent"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Step 4: Evaluation (Optional)
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Step 5: Save artifacts
joblib.dump(model, "intent_classifier.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("✅ Model and vectorizer saved successfully.")
