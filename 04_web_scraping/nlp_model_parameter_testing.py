# SCRIPT WITH COMMAND LINE INTERFACE TO TUNE THE HYPERPARAMETERS OF THE MODEL 

# to exectute it in the command line, type e.g.: python nlp_model.py -n_est=10 -max_d=2 -a=0.6

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


import sys
import argparse

parser = argparse.ArgumentParser(description='This program contains a NLP model that predicts if a given word relates either lyrics of the band Bastille or Rolling Blackouts Coastal Fever')

# positional arguments
#parser.add_argument("word", help="type the string you use here")

parser.add_argument("-n_est", "--n_estimators", help="Number of trees", type= int, default=20)
parser.add_argument("-max_d", "--max_depth", help="Maximum Depth of the Trees", type= int, default=5)
parser.add_argument("-a", "--alpha", help="alpha to smoothe Naive Bayes", type= float, default=1)


args = parser.parse_args()


df = pd.read_csv('data/df/artist_song.csv')

nlp = spacy.load('en_core_web_md')


# use the Spacy library to apply tokenization, 
# stemming or lemmatization when building your Bag Of Words feature matrix
def clean_text(corpus, model):
    """preprocess a string (tokens, stopwords, lowercase, lemma & stemming) returns the cleaned result
        params: review - a string
                model - a spacy model
                
        returns: list of cleaned strings
    """
    
    new_doc = []
    doc = model(corpus)
    for word in doc:
        if not word.is_stop and word.is_alpha:
            new_doc.append(word.lemma_.lower())
    
    cleaned_string = ", ".join(new_doc)  # putting the strings back into one string
    return cleaned_string


# apply function to each row of the song column in the dataframe 
df['song_spacy'] = df['lyrics'].apply(clean_text, args=(nlp,)) 


X = df['song_spacy']
y = df['artist']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # stratify=y

cv = TfidfVectorizer(stop_words='english')

new_corpus_x_train = cv.fit(X_train).transform(X_train)
new_corpus_x_test = cv.fit(X_train).transform(X_test)

df_cv_x_train = pd.DataFrame(new_corpus_x_train.todense(), columns=cv.get_feature_names()) 
df_cv_x_test = pd.DataFrame(new_corpus_x_test.todense(), columns=cv.get_feature_names()) # , index=['coldplay', 'masego']

# join y train and y test respectively 
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

df_cv_x_train['y_train'] = y_train
df_cv_x_test['y_test'] = y_test

df_train = df_cv_x_train
df_test = df_cv_x_test

X_train = df_train.iloc[:,:-1]
y_train = df_train.y_train

X_test = df_test.iloc[:,:-1]
y_test = df_test.y_test


# RandomOverSampler Model
ros = RandomOverSampler(sampling_strategy={'rolling_blackouts_coastal_fever':64})
X_ros, y_ros = ros.fit_resample(X_train, y_train)

# Fit the RandomOverSampling, 
rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=10)
rf.fit(X_ros, y_ros)

# Naive Bayes MultinomialNB
nb = MultinomialNB(alpha=args.alpha) # smaller alphas = way better results - but maybe overfitted?
nb.fit(X_train, y_train)
ypred_nb = nb.predict(X_test)

#print(rf.predict(args.word))

def print_evaluations(ytrue, ypred, model):
    """ Prints evaluation metrics for a specified model."""

    print(f'How does model {model} score:')
    print(f'The accuracy of the model is: {round(accuracy_score(ytrue, ypred), 3)}')
    print(f'The precision of the model is: {round(precision_score(ytrue, ypred, pos_label="bastille_" ), 3)}')
    print(f'The recall of the model is: {round(recall_score(ytrue, ypred, pos_label="bastille_"), 3)}')
    print(f'The f1-score of the model is: {round(f1_score(ytrue, ypred, pos_label="bastille_"), 3)}')
    
# Make predictions on the test data
ypred_rf = rf.predict(X_test)

# inspect metrics
print_evaluations(y_test.values, ypred_rf, 'RandomForest')
print_evaluations(y_test.values, ypred_nb, 'NaiveBayes')

