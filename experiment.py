from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import RandomizedSearchCV

from classify_documents_model.pipeline import text_clf

categories = ['talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc']

twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

parameters = {
    'tokens__lemmatization': (True, False),
    'tokens__remove_stopwords': (True, False)
}

X = twenty_train.data[:400]
y = twenty_train.target[:400]

clf  = RandomizedSearchCV(text_clf, parameters, cv=5, iid=False, 
                    scoring="precision_macro",
                    n_iter=3, n_jobs=-1, verbose=3)
clf.fit(X, y)

print("Mean test score: {}".format(clf.cv_results_["mean_test_score"]))
print("remove_stopwords: {}".format(clf.cv_results_["param_tokens__remove_stopwords"]))
print("lemmatization: {}".format(clf.cv_results_["param_tokens__lemmatization"]))
print("Best params: {}".format(clf.best_params_))
