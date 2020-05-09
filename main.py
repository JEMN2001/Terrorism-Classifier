import pandas as pd
import numpy as np
import math as mt

##########################################################################################
################################ Data import ###########################################
##########################################################################################
'''
Important: Before run this code, be sure that data have passed first filter
'''
def read_data(filename):
    data = pd.read_csv(filename, sep=',')
    return data

per_to_val = 0.1 #Percantage of the data used for the validation

train = read_data('First_filter/terrorism_filtered.csv') #terrorism.csv contains the ISIS related tweets
num_sample = len(train['tweets']) #num_sample represents the total amount of tweets this far


#Give this tweets the tag terrorism
temp_tag = ['terrorism related' for i in range(num_sample)]
train['tag'] = temp_tag

data_train = mt.ceil(num_sample*(1-per_to_val))
train, validate = train[:data_train], train[data_train:]


#Now we add the Fanboys database
tmp = read_data('First_filter/IsisFanboy_filtered.csv')

num_tmp = len(tmp['tweets'])
temp_tag = ['terrorism related' for i in range(num_tmp)]
tmp['tag'] = temp_tag

data_train = mt.ceil(num_tmp*(1-per_to_val))
tmp_train, tmp_validate = tmp[:data_train], tmp[data_train:]
train = pd.concat([train,tmp_train])
validate = pd.concat([validate,tmp_validate])

#Now we add the About ISIS database
tmp = read_data('First_filter/AboutIsis_filtered.csv')
tmp = tmp[:len(train)+len(validate)]

num_tmp = len(tmp['tweets'])
temp_tag = ['no terrorist' for i in range(num_tmp)]
tmp['tag'] = temp_tag

data_train = mt.ceil(num_tmp*(1-per_to_val))
tmp_train, tmp_validate = tmp[:data_train], tmp[data_train:]
train = pd.concat([train,tmp_train])
validate = pd.concat([validate,tmp_validate])

print('Number of tweets for train: ',len(train),'\nNumber of tweets for validate: ', len(validate))
##########################################################################################
################################ Data cleaning ###########################################
##########################################################################################

print("Cleaning the data")

tweets_train, tags_train = train['tweets'].values, train['tag'].values
tweets_train, tags_train = [str(x) for x in tweets_train], [str(x) for x in tags_train]

tweets_val, tags_val = validate['tweets'].values, validate['tag'].values
tweets_val, tags_val = [str(x) for x in tweets_val], [str(x) for x in tags_val]

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re

REMOVE_URLS = re.compile('http\S+')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string

        return: modified initial string in lowercase, with no URLs or unwanted symbols or english stopwords
    """
    text = text.replace("ENGLISH TRANSLATION: ","")
    text = text.lower()
    text = re.sub(REMOVE_URLS, "", text)
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)
    text = re.sub(BAD_SYMBOLS_RE,"",text)
    text = text.split();
    return ' '.join([i for i in text if i not in STOPWORDS])


tweets_train = [text_prepare(x) for x in tweets_train]
tweets_val = [text_prepare(x) for x in tweets_val]
##############################################################################################################
print("All tweets were cleaned")
##############################################################################################################
################################### Implementation of BagOfWords Strategy ####################################
##############################################################################################################

print("Implementing Bag of Words")
from collections import defaultdict
# Dictionary of all tags from train corpus with their counts.
tags_counts =  defaultdict(int)
# Dictionary of all words from train corpus with their counts.
words_counts =  defaultdict(int)


for tweet in tweets_train:
    for word in tweet.split():
        words_counts[word] += 1


for tag in tags_train:
        tags_counts[tag] += 1

print(tags_counts)

most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:6000]
DICT_SIZE = 5000
WORDS_TO_INDEX = {p[0]:i for i,p in enumerate(most_common_words[:DICT_SIZE])}
INDEX_TO_WORDS = {WORDS_TO_INDEX[k]:k for k in WORDS_TO_INDEX}
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """

    result_vector = np.zeros(dict_size)
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1

    return result_vector

from scipy import sparse as sp_sparse

tweets_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in tweets_train])
tweets_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in tweets_val])
print('tweets_train shape ', tweets_train_mybag.shape)
print('tweets_val shape ', tweets_val_mybag.shape)
print("My bag of Words transformation done!")

##############################################################################################################
################################### Implementation of TFIDF Strategy #########################################
##############################################################################################################
print("Implementing TFIDF")
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_val=tfidf_vectorizer.transform(X_val)
    return X_train, X_val, tfidf_vectorizer.vocabulary_

tweets_train_tfidf,tweets_val_tfidf, tfidf_vocab = tfidf_features(tweets_train,tweets_val)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
print("TFIDF transformation done!")

##############################################################################################################
########################################## Vectorizing the tags ##############################################
##############################################################################################################
print("Doing the MultiLabelBinarizer")
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
tags_train = mlb.fit_transform([[tag] for tag in tags_train])
tags_val = mlb.fit_transform([[tag] for tag in tags_val])

print("The MultiLabelBinarizer is done!")

##############################################################################################################
########################################## Training the model ################################################
##############################################################################################################
print("Training the model")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition.nmf import NMF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

def train_classifier(X_train, y_train, C=1.0, penalty='l2'):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    lr = LogisticRegression(solver='newton-cg',C=C, penalty=penalty,n_jobs=-1)
    # lr.fit(X_train, y_train)
    ovr = OneVsRestClassifier(lr)
    ovr.fit(X_train, y_train)
    return ovr

classifier_mybag = train_classifier(tweets_train_mybag, tags_train)
classifier_tfidf = train_classifier(tweets_train_tfidf, tags_train)

print("Both models are trained")

from joblib import dump, load
dump(classifier_mybag, 'Mybag_model.joblib')
dump(classifier_tfidf, 'Tfidf_model.joblib')

tags_val_predicted_labels_mybag = classifier_mybag.predict(tweets_val_mybag)
tags_val_predicted_scores_mybag = classifier_mybag.decision_function(tweets_val_mybag)

tags_val_predicted_labels_tfidf = classifier_tfidf.predict(tweets_val_tfidf)
tags_val_predicted_scores_tfidf = classifier_tfidf.decision_function(tweets_val_tfidf)

tags_val_pred_inversed = mlb.inverse_transform(tags_val_predicted_labels_tfidf)
tags_val_inversed = mlb.inverse_transform(tags_val)
for i in range(12,16):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        tweets_val[i],
        ','.join(tags_val_inversed[i]),
        ','.join(tags_val_pred_inversed[i])
    ))

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


def print_evaluation_scores(y_val, predicted):

    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
    print(average_precision_score(y_val, predicted))


print('Bag-of-words')
print_evaluation_scores(tags_val, tags_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(tags_val, tags_val_predicted_labels_tfidf)

from metrics import roc_auc

n_classes = len(tags_counts)
roc_auc(tags_val, tags_val_predicted_scores_mybag, n_classes)


n_classes = len(tags_counts)
roc_auc(tags_val, tags_val_predicted_scores_tfidf, n_classes)
