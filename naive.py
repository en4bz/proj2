import nltk, operator, sqlite3, math, sys
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import csv 

use_stemming = False

db = sqlite3.connect('comp598.db')

stop = stopwords.words('english')
stop += ['com','http','paper','result','also','one','two','three','use','new','show']

stop = set(stop)

CLASSES = ['cs','math','stat','physics']
REGEX = r'[A-Za-z]{3,}'

tokenizer = RegexpTokenizer(REGEX)
stemmer = SnowballStemmer('english')

def token(doc):
    if use_stemming:
        entry = set(stemmer.stem(w.lower()) for w in tokenizer.tokenize(doc) if w.lower())
    else:
        entry = set(w.lower() for w in tokenizer.tokenize(doc) if w.lower())
    return entry - stop


class NaiveBayes(object):
    all_data = db.execute('SELECT * from abstracts').fetchall()

    abstracts = [d[2] for d in all_data]
    targets = [d[1] for d in all_data]

    def __init__(self, num_features):
        self.num_feat = num_features

    def train(self, data, targets):

        doc_count = len(targets)
        # Calculate priors
        self.priors = {}
        for category in CLASSES:
            count = len([x for x in targets if x == category])
            self.priors[category] = count / float(doc_count)

        # Count word counts
        self.global_count = NaiveBayes.count_global(data=data)
        self.category_counts = {}
        for cat in CLASSES:
            cat_data = [tup[0] for tup in zip(data, targets) if tup[1] == cat]
            self.category_counts[cat] = self.count_category(cat, cat_data=cat_data)

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

    def predict(self, abstract):
        f_vec = self.to_feat_vec(abstract)
        score = -10900000000
        selected = None
        for cat in CLASSES:
            prob = self.calc_prob(cat, f_vec)
            if prob > score:
                score = prob
                selected = cat

        return selected

    def get_accuracy(self, test_abstracts, test_targets):
        results = []
        for (abst, tar) in zip(test_abstracts, test_targets):
            # import pdb; pdb.set_trace()
            if self.predict(abst) == tar:
                results.append(1)
            else:
                results.append(0)

        return float(sum(results)) / len(results)

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
    def count_global(data=None): 
        count = Counter()

        if data is None:
            data = db.execute('SELECT content FROM abstracts;').fetchall()
            data = [d[0] for d in data]

        for row in data:
            entry = list(token(row))
            count.update(entry)

        # print count.most_common(100)
        return count

    def count_category(self, cat, cat_data=None):
        count = Counter()
        tokenizer = RegexpTokenizer(REGEX)
        if cat_data is None:
            cat_data = db.execute('SELECT content FROM abstracts WHERE class = ?;', [cat]).fetchall()
            cat_data = [d[0] for d in cat_data]
        for row in cat_data:
            entry = list(token(row))
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

def print_results(num_features, test_start, test_end):
    classifier = NaiveBayes(num_features)
    train_data = classifier.abstracts[:test_start] + classifier.abstracts[test_end:]
    train_targets = classifier.targets[:test_start] + classifier.targets[test_end:]
    test_data = classifier.abstracts[test_start:test_end]
    test_targets = classifier.targets[test_start:test_end]

    classifier.train(train_data, train_targets)
    print "{0} {1} {2} {3} {4}".format(num_features,
                                   test_start,
                                   test_end,
                                   use_stemming,
                                   classifier.get_accuracy(test_data, test_targets)
                               )

def generate_test_predictions(test_input_file):
    classifier = NaiveBayes(2)
    classifier.train(classifier.abstracts, classifier.targets)
    preds = []

    with open(test_input_file) as f, open("output_brown_nb.csv", "w") as outf:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        writer = csv.writer(outf, delimiter=',', quotechar='"')
        reader.next()
        writer.writerow(["id", "category"])
        for row in reader:
            writer.writerow([row[0], classifier.predict(row[1])])

    print "Wrote ouput at output_brown_nb.csv"



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Not enough arguments!'
        sys.exit(2)

    if sys.argv[1] == 'getresults':
        try:
            num_features = int(sys.argv[2])
        except:
            print "Invalid second argument. Need an int: num_features"
            sys.exit(1)
        try:
            test_start = int(sys.argv[3])
            test_end = int(sys.argv[4])
        except:
            print "Third and fourth argument must be an int: test_start_index, test_end_index"
            sys.exit(1)

        if len(sys.argv) > 5 and sys.argv[5] == "true":
            use_stemming = True

        print_results(num_features, test_start, test_end)

    elif sys.argv[1] == 'generate_test_predictions':
        try:
            generate_test_predictions(sys.argv[2])
        except:
            print "Unkown error"
            print "Usage: python {0} generate_test_predictions <test_input_path>".format(sys.argv[0])
            raise

    else:
        classifier = NaiveBayes(3000)
        #print 'Train'
        classifier.train(classifier.abstracts[:500], classifier.targets[:500])
        print classifier.get_accuracy(classifier.abstracts[500:1000], classifier.targets[500:1000])
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
