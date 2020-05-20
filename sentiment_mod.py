# Sentiment Analysis Module
# Text classifier
# can be apply for any label text as long as they have to categories
# spam msg classifier
# Now we stop shuffling our data, now we know which is positive and which is negative.

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.classify import ClassifierI
from statistics import mode
import codecs
import re 

class voteclassifier(ClassifierI):

         def __init__(self, *classifiers):
                  self._classifiers = classifiers

         def classify(self, features):
                  votes = []
                  for c in self._classifiers:
                           v = c.classify(features)
                           votes.append(v)
                  return mode(votes)
                  
         def confidence(self, features):
                  votes = []
                  for c in self._classifiers:
                           v = c.classify(features)
                           votes.append(v)
                  choice_votes = votes.count(mode(votes))
                  conf = choice_votes / len(votes)
                  return conf



short_pos = codecs.open("Short_review/positive.txt","r", encoding='latin2').read()
short_neg = codecs.open("Short_review/negative.txt","r", encoding='latin2').read()

documents = []

all_words = []

'''

# j is adject, r is adverb and v is verb
# allowed_words_types = ["R","J", "V"]
allowed_word_types = ["J"]

for p in short_pos.split("\n"):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for n in  short_neg.split("\n"):
    documents.append((n, "neg"))
    words = word_tokenize(n)
    pos =  nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("pickle_files/documents", "wb")
pickle.dump(documents, save_documents)
save_documents.close()
'''

document_p = open("pickle_files/documents", "rb")
documents = pickle.load(document_p)
document_p.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

'''
save_feature  = open("pickle_files/word_features", "wb")
pickle.dump(word_features, save_feature)
save_feature.close()
'''
feature_p  = open("pickle_files/word_features", "rb")
word_features = pickle.load(feature_p)
feature_p.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
      features[w] = (w in words)
        

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

### NaiveBayesClassifier
##classifier_p = codecs.open("pickle_files/NaiveBayesClassifier", "rb")
##classifier = pickle.load(classifier_p.read())
##classifier_p.close()
##print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)



with open("pickle_files/NaiveBayesClassifier", 'rb') as file:  
    classifier = pickle.load(file)

with open("pickle_files/MNB_classifier", "rb") as file:
    MNB_classifier = pickle.load(file)


with open("pickle_files/BNB_classifier", "rb") as file:
    BNB_classifier = pickle.load(file)

with open("pickle_files/log_classifier", "rb") as file:
    log_classifier = pickle.load(file)

with open("pickle_files/SGD_classifier", "rb") as file:
    SGD_classifier = pickle.load(file)

#SVC_classifier_p = open("pickle_files/SVC_classifier", "rb")
#SVC_classifier = pickle.load(SVC_classifier_p)
#SVC_classifier_p.close()

with open("pickle_files/LinearSVC_classifier", "rb") as file:
    LinearSVC_classifier = pickle.load(file)

#NuSVC_classifier_p = open("pickle_files/NuSVC_classifier", "rb")
#NuSVC_classifier = pickle.load(NuSVC_classifier_p)
#NuSVC_classifier_p.close()


voted_classifier = voteclassifier(classifier,
                                  LinearSVC_classifier,
                                  SGD_classifier,
                                  log_classifier, BNB_classifier)


def sentiment(text):
    feats = find_features(text)

    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats) 











