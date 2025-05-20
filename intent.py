import joblib

# Load once at startup
intent_clf = joblib.load("intent_classifier.joblib")
intent_vectorizer = joblib.load("tfidf_vectorizer.joblib")

def predict_intent(question: str) -> str:
    cleaned = question.strip().lower()
    vec = intent_vectorizer.transform([cleaned])
    return intent_clf.predict(vec)[0]
