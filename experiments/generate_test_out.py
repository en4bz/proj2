import csv
import pickle
import numpy as np
import data_preparation as dp

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

data = np.array(dp.abstracts)
targets = np.array(dp.targets)

test_data = []
with open("../../data/test_input.csv") as csvfile:
    reader = csv.reader(csvfile)
    reader.next() # Skipping header line
    for row in reader:
        test_data.append(row[1])

class Unsparser(object):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X):
        return X.toarray()

svm_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.05)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier(loss="hinge",
                   penalty='l2', alpha=1e-3,
                   n_iter=5))
])

multi_nb_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.8,)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

rf_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.05, max_features=1000)),
                     ('tfidf', TfidfTransformer()),
                     ('unsparse', Unsparser()),
                     ('clf', RandomForestClassifier(n_estimators=30, n_jobs=-1)),
])

def generate_output(clf, clf_name):
    print "####################"
    print "Generating output for {0}".format(clf_name)
    print "Training classifier"
    clf.fit(data, targets)

    print "Pickling classifiers for later use"
    with open("pickled_{0}.pickle".format(clf_name), "wb") as f:
        pickle.dump(clf, f)

    print "Generating predictions"
    predicts = clf.predict(test_data)

    print "Writing out predicitons"
    with open("output_{0}.csv".format(clf_name), "w") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "category"])
        for i, cat_ind in enumerate(predicts):
            writer.writerow([i, dp.categories[cat_ind]])

generate_output(svm_clf, "SVM")
generate_output(multi_nb_clf, "multi_NB")
generate_output(rf_clf, "random_forest")
