sklearn_source_dir="/Users/droy/code/scikit-learn"

import sys
sys.path.insert(0, sklearn_source_dir)

import csv
import sqlite3
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.grid_search import GridSearchCV

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


db = sqlite3.connect("/Users/droy/Downloads/comp598.db")
raw_data = db.execute("SELECT content,class from abstracts").fetchall()
abstracts,classes = zip(*raw_data)

abstracts = list(abstracts)
classes = list(classes)

categories = []
for c in classes: 
    if c not in categories:
        categories.append(c)

assert sum([classes.count(x) for x in categories]) == len(classes)

targets = [categories.index(c) for c in classes]

train_test_ratio = 0.8
split_list = lambda l, frac: (l[:int(frac*len(l))],l[int(frac*len(l)):])

train_data, test_data = split_list(abstracts, train_test_ratio)
train_target, test_target = split_list(targets, train_test_ratio)

train_data = np.array(train_data)
train_target = np.array(train_target)
train_data = np.array(train_data)
train_target = np.array(train_target)



boost_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', AdaBoostClassifier(n_estimators=100)),
])

forest_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier()),
])

multi_nb_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.8, tokenizer=LemmaTokenizer())),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

svm_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.05)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier(loss="hinge",
                   penalty='l2', alpha=1e-3,
                   n_iter=5))
])

bern_nb_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.8)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB()),
])

chi2_svm_clf = Pipeline([('vect', CountVectorizer(stop_words="english", max_df=0.05)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier(loss="hinge",
                   penalty='l2', alpha=1e-3,
                   n_iter=5))
])


def score_me(raven, post_fits=[]):
    raven.fit(train_data, train_target)
    for fn in post_fits:
        fn(raven)
    preds = raven.predict(test_data)
    return np.mean(preds == test_target)

def dump_vocabulary(raven):
    vocs = raven.steps[0][1].vocabulary_
    vocs_list = vocs.keys()
    with open("vocs", "w") as f:
        for wrd in sorted(vocs_list):
            f.write(wrd + '\n')

def search_for_best_clf(raven):
    parameters = {'vect__max_df': [float(x)/100 for x in xrange(8,80,8)],
                'vect__tokenizer': [None, LemmaTokenizer()],
              'tfidf__use_idf': (True, False),
    }
    gs_clf = GridSearchCV(raven, parameters, n_jobs=-1)
    gs_clf.fit(train_data, train_target)
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
    print "Score: ", score



############################
print "All assertions passed!"