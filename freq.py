import nltk, operator, sqlite3, math, sys
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np


db = sqlite3.connect('comp598.db')

stop = stopwords.words('english')
stop += ['com','http','paper','results','also','one','two','three','used','new',
        'using','show']

stop = set(stop)

CLASSES = ['cs','math','stat','physics']

DOC_COUNT = float(db.execute('SELECT COUNT(*) FROM abstracts;').fetchone()[0])

PRIOR = {}
PRIOR['cs'] = db.execute("SELECT COUNT(*) FROM abstracts WHERE class = 'cs';").fetchone()[0] / DOC_COUNT
PRIOR['stat'] = db.execute("SELECT COUNT(*) FROM abstracts WHERE class = 'stat';").fetchone()[0] / DOC_COUNT
PRIOR['math'] = db.execute("SELECT COUNT(*) FROM abstracts WHERE class = 'math';").fetchone()[0] / DOC_COUNT 
PRIOR['physics'] = db.execute("SELECT COUNT(*) FROM abstracts WHERE class = 'physics';").fetchone()[0] / DOC_COUNT

NUM_FEAT = 1000

class NaiveBayes(object):

    SELECT_CLASS = "SELECT COUNT(*) FROM abstracts WHERE class = ?;"

    def __init__():
        self.priors = calc_priors()
        self.global_word_count  
        self.cls_dist = 

    @classmethod
    def calc_priors(cls):
        priors = {}
        priors['cs'] = db.execute(cls.SELECT_CLASS, ['cs']).fetchone()[0] / DOC_COUNT
        priors['stat'] = db.execute(cls.SELECT_CLASS, ['stat']).fetchone()[0] / DOC_COUNT
        priors['math'] = db.execute(cls.SELECT_CLASS, ['math']).fetchone()[0] / DOC_COUNT 
        priors['physics'] = db.execute(cls.SELECT_CLASS, ['physics']).fetchone()[0] / DOC_COUNT
        return priors

def distinct_words(cls):
    count = cls_count(cls)
    return reduce(lambda x,y: x + y[1], count.items(), 0)

def top_all():
    count = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    cur = db.execute('SELECT content FROM abstracts;')
    for row in cur.fetchall():
        entry = [w.lower() for w in tokenizer.tokenize(row[0]) if w.lower() not in stop and len(w) > 2]
        count.update(entry)

    return count

def cls_count(cls):
    count = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    cur = db.execute('SELECT content FROM abstracts WHERE class = ?;', [cls])
    for row in cur.fetchall():
        entry = [w.lower() for w in tokenizer.tokenize(row[0]) if w.lower() not in stop and len(w) > 2]
        count.update(entry)

    return count


def cls_dist(cls, features):
    cls_features = cls_count(cls)
    total = top_all()

    feature_dist = [0] * len(features)
    distinct_in_cls = float(distinct_words(cls))
    for i, word in enumerate(features):
        #feature_dist[i] = (cls_features[word] + 1) / float(total[word])
        feature_dist[i] = (cls_features[word] + 1) / distinct_in_cls 
    return feature_dist


def to_feat_vec(content, feature_set):
    tokenizer = RegexpTokenizer(r'\w+')
    entry = set(w.lower() for w in tokenizer.tokenize(content) if w.lower() not in stop and len(w) > 2)
    feat_vec = np.zeros(len(feature_set), np.int8)
    for i, word in enumerate(feature_set):
        if word in entry:
            feat_vec[i] = 1

    return feat_vec

def calc_prob(cls, features, dist):
    prod = math.log(PRIOR[cls])
    for i in range(0, len(features)):
        if features[i] > 0:
            prod += math.log(dist[i])
        else:
            prod += math.log(1 - dist[i])

    return prod

def calc_dists():
    all_dists = {}
    top_features = [x[0] for x in top_all().most_common(NUM_FEAT)]
    for cls in CLASSES:
        all_dists[cls] = cls_dist(cls, top_features)

    return all_dists


def test():
    all_dists = calc_dists()
    top_features = [x[0] for x in top_all().most_common(NUM_FEAT)]
    cur = db.execute('SELECT id,content FROM test;')
    for row in cur.fetchall():
        f_vec = to_feat_vec(row[1], top_features) 
        score = -1000000000
        selected  = ''
        for cls in CLASSES:
            prob = calc_prob(cls, f_vec, all_dists[cls])
            if prob > score:
                score = prob
                selected = cls

        print '"%s","%s"' % (row[0], selected)


if __name__ == '__main__':

#    top_features = [x[0] for x in top_all().most_common(1000)]
    test()
    sys.exit(0)
#    print 'TOP ALL', top_features


#    for i in range(0,100):
#        print all_dists['cs'][i] + all_dists['stat'][i] + all_dists['math'][i] + all_dists['physics'][i]


    cur = db.execute('SELECT id,class,content FROM abstracts LIMIT 500;')
    right = 0
    total = 0
    for row in cur.fetchall():
        f_vec = to_feat_vec(row[2], top_features) 
#        print top_features[1:10]
#        print f_vec[1:10]
#        print row[2]
#        print 'ACTUAL CLASS', row[1],
        score = -10900000000
        selected  = ''
        for cls in CLASSES:
            prob = calc_prob(cls, f_vec, all_dists[cls])
            if prob > score:
                score = prob
                selected = cls

        if selected == row[1]:
            right += 1

        total += 1
    
    print(right / float(total))
