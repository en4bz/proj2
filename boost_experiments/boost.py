import data_preparation as dp


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

class Unsparser(object):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X.toarray()

def main(max_features, n_estimators):

    boost_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.05, max_features=max_features)),
                         ('tfidf', TfidfTransformer()),
                         ('unsparse', Unsparser()),
                         ('clf', AdaBoostClassifier(n_estimators=n_estimators)),
    ])

    print dp.score_me(boost_clf)

if __name__ == "__main__":
    import sys
    maxf = int(sys.argv[1])
    n_est = int(sys.argv[2])
    main(maxf, n_est)