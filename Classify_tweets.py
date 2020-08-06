import pandas as pd
import numpy as np
import math as mt
from joblib import load

##########################################################################################
################################ Data import ###########################################
##########################################################################################
'''
Important: Before run this code, be sure that data have passed first filter and Model is ready to predict
'''
def read_data(filename):
    data = pd.read_csv(filename, sep=',')
    return data

file_name = 'Validation_tweets'
col_tweets = 'text'
sample = read_data('First_filter/'+file_name+'.csv') #terrorism.csv contains the ISIS related tweets


print('Number of tweets to categorize: ',len(sample))
##########################################################################################
################################ Data cleaning ###########################################
##########################################################################################

print("Cleaning the data")

tweets = sample[col_tweets].values
tweets = [str(x) for x in tweets]
tweets_display = tweets

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

import re

REMOVE_URLS = re.compile('http\S+')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|,;&\n]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +\'@#]')
STOPWORDS = set(stopwords.words('english'))
tokenizer = nltk.tokenize.WhitespaceTokenizer()
stemmer = nltk.stem.WordNetLemmatizer()

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
    text = tokenizer.tokenize(text);
    return ' '.join([stemmer.lemmatize(i) for i in text if i not in STOPWORDS])


tweets = [text_prepare(x) for x in tweets]
##############################################################################################################
print("All tweets were cleaned")
##############################################################################################################
################################### Implementation of BagOfWords Strategy ####################################
##############################################################################################################

print("Implementing Bag of Words")
from collections import defaultdict
# Dictionary of all words from train corpus with their counts.
words_counts =  defaultdict(int)

for tweet in tweets:
    for word in tweet.split():
        words_counts[word] += 1


most_common_words = load("Trained_Model/BagOfWords.joblib")
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

tweets_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in tweets])
print('tweets shape ', tweets_mybag.shape)
print("My bag of Words transformation done!")

##############################################################################################################
################################### Implementation of TFIDF Strategy #########################################
##############################################################################################################
print("Implementing TFIDF")
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_val):
    """
        X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    tfidf_vectorizer = load("Trained_Model/tfidf_vectorizer.joblib")
    X_val=tfidf_vectorizer.transform(X_val)
    return X_val, tfidf_vectorizer.vocabulary_

tweets_tfidf, tfidf_vocab = tfidf_features(tweets)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
print("TFIDF transformation done!")

##############################################################################################################
########################################## Vectorizing the tags ##############################################
##############################################################################################################
print("Doing the MultiLabelBinarizer")
from sklearn.preprocessing import MultiLabelBinarizer

mlb = load("Trained_Model/MultiLabelBinarizer.joblib")

print("The MultiLabelBinarizer is done!")

##############################################################################################################
########################################## Upload Training the model #########################################
##############################################################################################################
print("Uploading Trained model")

classifier_mybag = load('Trained_Model/Mybag_model.joblib')
classifier_tfidf = load('Trained_Model/Tfidf_model.joblib')

print("Both models are Uploaded")

tags_predicted_labels_mybag = classifier_mybag.predict(tweets_mybag)
tags_predicted_scores_mybag = classifier_mybag.decision_function(tweets_mybag)

tags_predicted_labels_tfidf = classifier_tfidf.predict(tweets_tfidf)
tags_predicted_scores_tfidf = classifier_tfidf.decision_function(tweets_tfidf)

tags_pred_inversed = mlb.inverse_transform(tags_predicted_labels_tfidf)
for i in range(95,100):
    print('Title:\t{}\nWithout NLP:\t{}\nPredicted labels:\t{}\n\n'.format(
        tweets[i],
        tweets_display[i],
        ','.join(tags_pred_inversed[i])
    ))

##############################################################################################################
########################################## Save the predictions ##############################################
##############################################################################################################
sample['text'] = tweets_display
sample['tag'] = [''.join(tags_pred_inversed[i]) for i in range(len(tweets_display))]
sample.to_csv('Classified_'+file_name+'.csv',index=False)
