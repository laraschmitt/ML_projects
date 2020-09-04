### NLP Model prediction after input


word = input('Please type in a word or lyrics: ')



import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, confusion_matrix, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# import preprocessed training data
df_preproceesed = pd.read_csv('data/df/df_artists_song_preprocessed.csv')

# define feature matrix and response vector
X = df_preproceesed['song_spacy']
y = df_preproceesed['artist']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # stratify=y

# Fit count vectorizer
cv = TfidfVectorizer(stop_words='english')
corpus = cv.fit(X_train)
new_corpus_x_train = cv.fit(X_train).transform(X_train)
new_corpus_x_test = cv.fit(X_train).transform(X_test)

df_train = pd.DataFrame(new_corpus_x_train.todense(), columns=cv.get_feature_names()) 
df_test = pd.DataFrame(new_corpus_x_test.todense(), columns=cv.get_feature_names()) # , index=['coldplay', 'masego']

# join y train and y test respectively 
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

df_train['y_train'] = y_train
df_test['y_test'] = y_test


X_train = df_train.iloc[:,:-1]
y_train = df_train.y_train

X_test = df_test.iloc[:,:-1]
y_test = df_test.y_test


# instatiate RandomForest Classifier and fit a model

rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=10)
# with oversampling
ros = RandomOverSampler(sampling_strategy={'rolling_blackouts_coastal_fever':64})
X_ros, y_ros = ros.fit_resample(X_train, y_train)

rf.fit(X_ros, y_ros)

# make Random Forest predictions for an unknow word
print('My Random Forest Classifier (n_estimators=30, max_depth=5) predicts that your word belongs to the wordspace of the following artist:')

cv = TfidfVectorizer(stop_words='english')
new= corpus.transform(['word'])

ypred_rf_single_word = rf.predict(new) # returns an array
output = ypred_rf_single_word[0]

print(output)

# Naive Bayes MultinomialNB
print('My Naive Bayes classifier predicts that your word belongs to the wordspace of the following artist:')

nb = MultinomialNB(alpha=0.3) # smaller alphas = way better results - but maybe overfitted?
nb.fit(X_train, y_train)

ypred_nb_single_word = nb.predict(new) # returns an array
output2 = ypred_nb_single_word[0]

print(output2)