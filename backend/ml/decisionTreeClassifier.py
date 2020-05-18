import nltk
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import random
from nltk.corpus import brown
import math
from operator import itemgetter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from confusion_matrix import getCM
import pickle
import os
from dtwo import saveD



# model_path = 'a.pkl'
# if not os.path.exists(model_path):

data = []
stop_words = set(stopwords.words('english')) #All meaningless words such as 'to', 'at'...

#basically split a sentence into words
def tokenize_raw_data(raw_data):
    print('tokenizing data')
    tokenized = []
    for row in raw_data:    #for each row in csvfile
        #only need to tokenize the joke column
        tokenized.append([row[0], row[1], row[2], word_tokenize(row[3])])
        #tokenized.append([row[0], row[1], row[2], word_tokenize(row[3]), row[4], row[5]])
    return tokenized

#the program starts here
with open('Final CSV  - Sheet1.csv') as file:
#with open('comedy_central.csv') as file:
        print('\n\n')
        reader = csv.reader(file, delimiter=',')
        line = 0
        print('Reading data from csv file')
        for row in reader:
            if line != 0:
                data.append(row)
            line += 1
print('Reading complete')
tokenized_data = tokenize_raw_data(data)
print('Training...')
total = 0
feature = set() #used set so that there will be no duplicate data
for counter in range(50):  #train and test 100 times
    #print(counter)
    random.shuffle(tokenized_data) #shuffle the data so that we get a more accurate result
    feature_sets = [] #actual feature set
    for i in tokenized_data:
        word_occur = {}
        for word in i[0]:
        #for word in i[5]:
            # since the classifier classifies all data into michelle wolf
            if word  != "":
                #validate
                for word in i[3]:
                    if word not in punctuation and word.lower() not in ['trump', 'obama',  'laughter', 'cheering', 'applause',] and word != '``' and word != "''" and word != '""' and word.lower() not in stop_words:

                        #create dictionary of (word, occurence)
                        if word not in word_occur:
                            word_occur.update(dict(word = 1))
                        else:
                            word_occur[word] = word_occur[word] + 1
            #uses occerence as feature
            feature_sets.append((word_occur, i[0]))
            #feature_sets.append((word_occur, i[5]))

    training_set = feature_sets[:80] #uses 50 joke to train the classifier
    testing_set= feature_sets[80:]  #uses the other to test result

    #set trained by decision tree classifier
    classifier = nltk.DecisionTreeClassifier.train(training_set, depth_cutoff=0.1, support_cutoff=500, entropy_cutoff=0.1)

    #get percentage of accuracy
    total += (nltk.classify.accuracy(classifier, testing_set))*100
print("Classifier accuracy percent:", total/50, '\n')

#     pickle.dump(classifier, open(model_path, 'wb' ))
# else:
#     classifier = pickle.load(open(model_path, 'rb'))

# var = input("Please enter something: ")
# print(classifier.classify(var))

#saveD(classifier)



#print(testing_set)
#predicted = classifier.classify_many(testing_set)

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                        title='Normalized confusion matrix')

# plt.show()

#print confusion matrix
getCM(classifier, testing_set)
