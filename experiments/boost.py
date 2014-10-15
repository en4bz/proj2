import data_preparation as dp

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree


class Unsparser(object):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X.toarray()

class BoostClassifier(object):

    weight_factor = 100000000
    alphas = []
    clfs = []

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def new_clf(self):
        return tree.DecisionTreeClassifier


    def fit(self, X, y, total_classes):
        self.weights = np.ones(len(y), dtype=np.float64) * (1.0  / len(y))
        self.countvectorizer = CountVectorizer(stop_words="english", max_df=0.8)
        cv_X = self.countvectorizer.fit_transform(X, y)
        self.tfidf = TfidfTransformer()
        self.X = self.tfidf.fit_transform(cv_X, y)
        self.y = y
        self.total_classes = total_classes
        for i in xrange(self.n_estimators):
            c = self.train_new_clf(self.weights)
            preds = c.predict(self.X)
            wrongs = ( preds != self.y)
            err = np.sum(self.weights * wrongs) / np.sum(self.weights)
            self.alphas.append(np.log((1 - err) / err) + np.log(total_classes - 1))
            self.clfs.append(c)
            self.weights = self.weights * np.exp(self.alphas[i] * wrongs)

        print self.score(c, self.X, self.y)
        print self.validation_score(c)
        print self.predict(self.X[0])
        print self.boost_score(X[:10], y[:10])
        print self.boost_score(dp.test_data[:100], dp.test_target[:100])

    def train_new_clf(self, weights):
        clf = self.new_clf()
        clf.fit(self.X, self.y, sample_weight=weights * self.weight_factor)
        return clf

    def normalize_weights(self, w):
        return w / np.sum(w)

    def score(self, clf, data, target, post_fits=[]):
        preds = clf.predict(data)
        return np.mean(preds == target)

    def validation_score(self, clf):
        d = self.countvectorizer.transform(dp.test_data)
        d = self.tfidf.transform(d)
        return self.score(clf, d, dp.test_target)

    def boost_score(self, X, y):
        X = self.countvectorizer.transform(X)
        X = self.tfidf.transform(X)

        preds = map(self.predict, X)

        preds = np.array(preds)
        return np.mean(preds == y)

        

    def predict(self, d):
        # d = self.countvectorizer.transform(x)
        # d = self.tfidf.transform(d)
        max_class = 0;
        max_score = - np.inf

        # alphas = np.array(self.alphas)
        # base_preds = np.array()

        for i in xrange(self.total_classes):
            score = 0
            for alpha, clf in zip(self.alphas, self.clfs):
                if clf.predict(d) == i:
                    score += alpha
            if score > max_score:
                max_score = score
                max_class = i
        return max_class




def main():

    bclf = BoostClassifier(20)
    bclf.fit(dp.train_data, dp.train_target, 4)
    # from IPython import embed; embed()


if __name__ == "__main__":
    main()