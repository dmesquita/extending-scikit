from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from .transformers import TextTransformer

text_clf = Pipeline([
    ('tokens', TextTransformer("en_core_web_sm")),
    ('tfidf', TfidfVectorizer(tokenizer=lambda x:x, lowercase=False)),
    ('clf', RandomForestClassifier(n_estimators=100)),
])
