import csv
from textblob.classifiers import NaiveBayesClassifier
import sqlite3

db = sqlite3.connect("/Users/droy/Downloads/comp598.db")
data = db.execute("SELECT content,class from abstracts").fetchall()

def get_from_csv(filename):
    data = []
    with open(filename, 'rb') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        data_reader.next()
        for row in data_reader:
            data.append(row)
    return data
    

train = data[0:50]
test = data[50:100]

# print "Training"
# cl = NaiveBayesClassifier(train)
# print "Testing"
# print cl.accuracy(test)


#########################

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

train_data = [x[0] for x in train]
train_target = [x[1] for x in train]

test_data = [x[0] for x in test]
test_target = [x[1] for x in test]
y_pred = gnb.fit(train_data, train_target).predict(test_data)
print("Number of mislabeled points out of a total %d points : %d"
       % (test_data.shape[0],(test_target != y_pred).sum()))