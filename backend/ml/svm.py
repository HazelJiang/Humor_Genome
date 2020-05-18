import os
from sklearn.svm import SVC #SVM model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #gives tfidf value for each word
from sklearn.model_selection import train_test_split #splits data
from nltk.stem import PorterStemmer #stems words
from nltk.tokenize import sent_tokenize, word_tokenize #sentence and words can be separated
from sklearn.pipeline import Pipeline #steps can be run in series
from sklearn.feature_extraction.text import CountVectorizer #converts text into number vectors
from sklearn.feature_extraction.text import TfidfTransformer #transforms count vectors into tfidf

#Reading and preprocessing
path = 'textfiles/Final CSV  - Sheet1.csv'
data1 = pd.read_csv(path)

comedians = data1.Comedian.tolist() #comedians
jokes = data1.Joke.tolist() #jokes
for i in range(len(jokes)): #jokes are made lower case
    jokes[i] = jokes[i].lower()

######## attempts to explore word stemming

#porter stemmer model
#ps = PorterStemmer()

# attempt to remove more stop words

# for sent_ind in range(len(jokes)):
#     if word_tokenize(jokes[sent_ind]) not in ['obama', 'laughter', 'trump', 'cheering', 'applause']:
#         jokes[sent_ind] = word_tokenize(jokes[sent_ind])

# pipeline to transform words into tf - idf vectors, then trains SVM model
model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])


# #tfid model
# tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2',
#     encoding='latin-1', ngram_range=(1, 2), stop_words='english') #assign tfids to words
# features = tfidf.fit_transform(jokes).toarray()

#splits into testing and training for both data and labels
X_train, X_test, y_train, y_test = train_test_split(jokes, comedians, test_size=0.3)

#trains model
model.fit(X_train, y_train)
print('training model ... ')
#prints accuracy score when tested against testing data and labels
print('accuracy: ' + str(model.score(X_test, y_test)))

#SVC model
# clf = SVC()
# clf.fit(X_train, y_train)
#print(clf.predict(X_test))

#accuracy score
#print('accuracy score: ' + str(clf.score(X_test, y_test))) #33.3% accuracy
