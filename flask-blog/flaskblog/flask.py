from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score
import six.moves.cPickle as pickle
from flaskblog.imdbReview import extract_words
import pandas as  pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV

with open('vect.pkl','rb') as f:
    vectorizer=pickle.load(f)

with open('funny.pkl','rb') as f:
    classifier_liblinear=pickle.load(f) 
    
input_features = vectorizer.transform(extract_words(sentences))
prediction = classifier_liblinear.predict(input_features)       