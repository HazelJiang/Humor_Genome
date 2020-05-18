#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #splits data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.pipeline import Pipeline #steps can be run in series
from sklearn.feature_extraction.text import CountVectorizer #converts text into number vectors
from sklearn.feature_extraction.text import TfidfTransformer #transforms count vectors into tfidf
from sklearn.svm import SVC #SVM model
import matplotlib.pyplot as plt

path = 'comedy_central.csv'
data = pd.read_csv(path)
joke_content = data.content.tolist() #actual joke
categories = data.flattened_categories.tolist()

#creates list of most relevant category label for each joke
categories_1 = []
for comma_categ in categories:
    categ_split = comma_categ.split(',') #splits into list of categories
    categories_1.append(categ_split[0]) #appends 1st (most relevant) category

#list of unique most relevant categories of all jokes
unique_categ = []
for categ in categories_1:
    if categ not in unique_categ:
        unique_categ.append(categ)

#cleans jokes of stopwords, punctuation, and irrelevant info
print('Cleaning jokes ...')
stop_words = set(stopwords.words('english')) #set of predefined english stop words
joke_content_cleaned = []
for sent in joke_content:
    words = word_tokenize(sent) #splits into words
    #print('joke content before being cleaned: ', words)
    words_cleaned = [] #joke, list of meaningful words
    for word in words:
        if word not in stop_words and word not in punctuation and word not in ['Q', 'A', '``']:
            words_cleaned.append(word)
    words_cleaned = ' '.join(words_cleaned)
    #print('joke content after being cleaned: ', words_cleaned)
    joke_content_cleaned.append(words_cleaned.strip())


# pipeline to transform words into tf-idf vectors, then trains SVM model
print('Creating pipeline ...')
model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(kernel='rbf'))])

#splits jokes and categories into testing and training sets
print('Splitting into testing and training sets ...')
X_train, X_test, y_train, y_test = train_test_split(joke_content_cleaned, categories_1, test_size=0.1)

#trains model
print('Training model ...')
model.fit(X_train, y_train)
#prints accuracy score when tested against testing data and labels
print('Accuracy: ', model.score(X_test, y_test))


















