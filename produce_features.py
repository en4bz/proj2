from sklearn.feature_extraction.text import CountVectorizer                     
import numpy as np                                                              
import sqlite3                                                                  
                                                                               
db = sqlite3.connect('/Users/droy/Downloads/comp598.db')                                              
                                                                               
MAX = 2 ** 12                                                                   
                                                                               
train_corpus = [x[0] for x in db.execute("SELECT content FROM abstracts ORDER BY id ASC;").fetchall()]
# test_corpus = [x[0] for x in db.execute("SELECT content FROM test ORDER BY id ASC;").fetchall()]
                                                                               
train = CountVectorizer(token_pattern=r'[A-Za-z]{3,}',max_features=MAX,stop_words='english',binary=True)
X = train.fit_transform(train_corpus)                                           
np.savetxt('features.txt', X.toarray(), '%d', '')                                 
                                                                               
                                                                               
# vocab = { word: i for  i, word in enumerate(train.get_feature_names()) }        
# print vocab                                                                     
                                                                               
# test = CountVectorizer(token_pattern=r'[A-Za-z]{3,}',stop_words='english',binary=True,vocabulary=vocab)
# Y = test.fit_transform(test_corpus)                                             
# np.savetxt('sk_16k.test', Y.toarray(), '%d', '')