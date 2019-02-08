# -*- coding: utf-8 -*-
import csv
import nltk
import numpy as np
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.sentiment.util import mark_negation
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas

#Preprocessor - Creating Dataset

stop_words = set(stopwords.words('english'))

#Requires the review.csv file to run

f = open('review.csv', "r", encoding = "utf-8")
read = csv.reader(f)
counter = 0
text_reviews = []
stars = []
for row in read:
    counter += 1
    if(counter < 200000): 
        if row[5] != '3':
            stars.append(row[5])
            text_reviews.append(row[6])
    else:
        f.close()
        text_reviews = text_reviews[1:]
        stars = stars[1:]
        break

# Preprocessor - Creating "Balanced Subset"

n = Counter(stars)
max_ = n.most_common()[-1][1]
n_added = {class_: 0 for class_ in n.keys()}
new_ys = []
new_xs = []
for i, y in enumerate(stars):
    if n_added[y] < max_:
        new_ys.append(y)
        new_xs.append(text_reviews[i])
        n_added[y] += 1
        
stars = new_ys
text_reviews = new_xs

#Preprocessor - Lemmatizer

xs_dupe = []
l = WordNetLemmatizer()
for x in text_reviews:
    review = str()
    for w in word_tokenize(x):
        current = str(l.lemmatize(w, pos = 'n'))
        for t in [l.lemmatize(w, pos = 'v'), l.lemmatize(w, pos = 'a'), l.lemmatize(w, pos = 'r')]:
            if len(t) < len(current):
                current = t
        if '\'' in current or '!' in current or '.' in current or ',' in current:
            review = review + current
        else:
            review = review + ' ' + current
    xs_dupe.append(review)
text_reviews = xs_dupe

#Vectorizer - n-grams, removing stopwords, negation detection

vectorizer = CountVectorizer(ngram_range=(1,3), stop_words = 'english', tokenizer=lambda text: mark_negation(word_tokenize(text)))
vectors = vectorizer.fit_transform(text_reviews)
stars = np.array(stars)

#5-Fold cross validation, training classifier, evaluating accuracy

kf = KFold(n_splits = 5)
kf.get_n_splits(vectors)
binary_accuracy_list = list()
classifier_accuracy_list = list()
for x,y in kf.split(vectors):
    x = list(x)
    y = list(y)
    X_train, X_test = vectors[x], vectors[y]
    y_train, y_test = stars[x], stars[y]
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    correct = 0
    for i, x in enumerate(y_test):
        if x == "1":
            if y_pred[i] == "1":
                correct += 1
            elif y_pred[i] == "2":
                correct += 1
        elif x == "2":
            if y_pred[i] == "1":
                correct += 1
            elif y_pred[i] == "2":
                correct += 1
        if x == "4":
            if y_pred[i] == "4":
                correct += 1
            elif y_pred[i] == "5":
                correct += 1
        if x == "5":
            if y_pred[i] == "4":
                correct += 1
            elif y_pred[i] == "5":
                correct += 1
    binary_accuracy_list.append(correct/len(y_test))
    classifier_accuracy_list.append(metrics.accuracy_score(y_test, y_pred))
classifier_accuracy_list = np.array(classifier_accuracy_list)
binary_accuracy_list = np.array(binary_accuracy_list)
print("4-class classification = ", classifier_accuracy_list.mean())
print("binary = ", binary_accuracy_list.mean())