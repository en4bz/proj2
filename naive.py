import nltk, operator, sqlite3, math, sys
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import numpy as np


db = sqlite3.connect('comp598.db')

stop = stopwords.words('english')
stop += ['com','http','paper','result','also','one','two','three','use','new','show']

stop = set(stop)

CLASSES = ['cs','math','stat','physics']
REGEX = r'[A-Za-z]{3,}'

tokenizer = RegexpTokenizer(REGEX)
stemmer = SnowballStemmer('english')

def token(doc):
    entry = set(stemmer.stem(w.lower()) for w in tokenizer.tokenize(doc) if w.lower())
    return entry - stop


class NaiveBayes(object):

    DOC_COUNT = db.execute('SELECT COUNT(*) FROM abstracts;').fetchone()[0]
    PRIOR_QUERY = "SELECT COUNT(*) FROM abstracts WHERE class = ?;"

    def __init__(self, num_features):
        self.num_feat = num_features

    def train(self):
        # Calculate priors
        self.priors = {}
        for category in CLASSES:
            cur = db.execute(NaiveBayes.PRIOR_QUERY, [category])
            count = cur.fetchone()[0]
            self.priors[category] = count / float(NaiveBayes.DOC_COUNT)

        # Count word counts
        self.global_count = NaiveBayes.count_global()
        self.category_counts = {}
        for cat in CLASSES:
            self.category_counts[cat] = self.count_category(cat)

        # Count total word counts for each class
        self.cat_totals = {}
        for cat in CLASSES:
            self.cat_totals[cat] = reduce(lambda x, y: x + y, self.category_counts[cat].values(), 0)

        # Select top global features
        self.top = [x[0] for x in self.global_count.most_common(self.num_feat)]

        # Calculate the Bernouli distributions over each feature for each
        # category
        self.dists = {}
        for cat in CLASSES:
            self.dists[cat] = np.zeros(self.num_feat, np.float64) 
            for i, word in enumerate(self.top):
                #feature_dist[i] = (cls_features[word] + 1) / float(total[word])
                self.dists[cat][i] = (self.category_counts[cat][word] + 1) / float(self.cat_totals[cat])

    def validate(self):
        cur = db.execute('SELECT id,class,content FROM abstracts LIMIT 1000;')
        right = 0
        total = 0
        for row in cur.fetchall():
            f_vec = self.to_feat_vec(row[2]) 
            score = -10900000000
            selected  = ''
            for cat in CLASSES:
                prob = self.calc_prob(cat, f_vec)
                if prob > score:
                    score = prob
                    selected = cat

            if selected == row[1]:
                right += 1

            total += 1
        
        print(right / float(total))
        return right / float(total)

    def write_features(self):
        cur = db.execute('SELECT id,class,content FROM abstracts;')
        for row in cur.fetchall():
            sys.stdout.write(row[1])
            sys.stdout.write(' ')
            entry = token(row[2])
            for i, word in enumerate(self.top):
                if word in entry:
                    sys.stdout.write('1')
                else:
                    sys.stdout.write('0')
            sys.stdout.write('\n')

    def write_test_features(self):
        cur = db.execute('SELECT id, content FROM test ORDER BY id;')
        for row in cur.fetchall():
            sys.stdout.write(str(row[0]))
            sys.stdout.write(' ')
            tokenizer = RegexpTokenizer(REGEX)
            entry = tocken(row[1])
            for i, word in enumerate(self.top):
                if word in entry:
                    sys.stdout.write('1')
                else:
                    sys.stdout.write('0')
            sys.stdout.write('\n')

    def test(self):
        FMT ='"%s","%s"'
        print FMT % ('id', 'category')
        cur = db.execute('SELECT id,content FROM test;')
        for row in cur.fetchall():
            f_vec = self.to_feat_vec(row[1]) 
            score = -1000000000
            selected  = ''
            for cls in CLASSES:
                prob = self.calc_prob(cls, f_vec)
                if prob > score:
                    score = prob
                    selected = cls

            print FMT % (row[0], selected)


    @staticmethod
    def count_global(): 
        count = Counter()
        cur = db.execute('SELECT content FROM abstracts;')
        for row in cur.fetchall():
            entry = list(token(row[0]))
            count.update(entry)

        print count.most_common(100)
        return count

    def count_category(self, cat):
        count = Counter()
        tokenizer = RegexpTokenizer(REGEX)
        cur = db.execute('SELECT content FROM abstracts WHERE class = ?;', [cat])
        for row in cur.fetchall():
            entry = list(token(row[0]))
            count.update(entry)

        return count

    def to_feat_vec(self, content):
        tokenizer = RegexpTokenizer(REGEX)
        entry = token(content)
        feat_vec = np.zeros(self.num_feat, np.int8)
        for i, word in enumerate(self.top):
            if word in entry:
                feat_vec[i] = 1

        return feat_vec

    def calc_prob(self, category, feat_vec):
        prod = math.log(self.priors[category])
        for i, word in enumerate(self.top):
            if feat_vec[i] > 0:
                prod += math.log(self.dists[category][i])
            else:
                prod += math.log(1 - self.dists[category][i])

        return prod

    # END NaiveBayes Class

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Not enough arguments!'
        sys.exit(2)

    classifier = NaiveBayes(3000)
    #print 'Train'
    classifier.train()
    #print 'Validate'
    if sys.argv[1] == 'test':
        classifier.test()
    elif sys.argv[1] == 'validate':
        classifier.validate()
    elif sys.argv[1] == 'features':
        classifier.write_features()
    elif sys.argv[1] == 'test_features':
        classifier.write_test_features()
    else:
        print 'Invalid option!'
        sys.exit(2)

    sys.exit(0)
