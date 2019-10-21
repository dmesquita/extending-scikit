import spacy

from sklearn.base import BaseEstimator, TransformerMixin

class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp_model="en", lemmatization=True, remove_stopwords=True) -> None:  
        self.nlp_model = nlp_model
        self.lemmatization = lemmatization
        self.remove_stopwords = remove_stopwords

    def fit(self, X: list, y=None) -> 'TextTransformer':
        return self

    def transform(self, X: list) -> list:
        model_ = spacy.load(self.nlp_model)
        X_ = []
        X_docs = list(model_.pipe(X, disable=["parser","ner"]))  
        for doc in X_docs:
            tokens = []
            if self.lemmatization:
                if self.remove_stopwords:
                    tokens = [t.lemma_ for t in doc if not t.is_stop]
                else:
                    tokens = [t.lemma_ for t in doc]
            else:
                if self.remove_stopwords:
                    tokens = [t.lower_ for t in doc if not t.is_stop]
                else:
                    tokens = [t.lower_ for t in doc]
                
            X_.append(tokens)

        return X_

