from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sqlite3

db = sqlite3.connect('comp598.db')
CLASSES = ['cs','math','stat','physics']

MAX = 2 ** 12 + 1000

corpus = [x[0] for x in db.execute("SELECT content FROM abstracts;").fetchall()]

words = {}
vecs = {}
for cls in CLASSES:
    words[cls] = [x[0] for x in db.execute("SELECT content FROM abstracts WHERE class = ?;",[cls]).fetchall()]
    vecs[cls] = CountVectorizer(token_pattern=r'[A-Za-z]{3,}',max_features=MAX,stop_words='english')
    vecs[cls].fit(words[cls])


fset = set()
for cls in CLASSES:
    forcls = set(vecs[cls].get_feature_names())
    for comp in CLASSES:
        if cls == comp:
            continue
        else:
            forcls -= set(vecs[comp].get_feature_names())

    fset |= forcls


fset.pop()
print len(fset)
exit(0)

fmap = {}
for i in range(0,2048):
    fmap[fset.pop()] = i


count = CountVectorizer(token_pattern=r'[A-Za-z]{3,}',max_features=MAX,stop_words='english',vocabulary=fmap,binary=True)
X = count.fit_transform(corpus)
np.savetxt('new_feat_2048.txt',X.toarray(),'%d','')
