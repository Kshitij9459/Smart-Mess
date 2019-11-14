from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score
import six.moves.cPickle as pickle
from imdbReview import extract_words
import pandas as  pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV
# Load All Reviews in train and test datasets

df=pd.read_csv('train.csv')
reviews=np.array(df.loc[:,'text'])
test=reviews[9000:]
y_train=np.array(df.loc[:,'funny'])
y_test=y_train[9000:]
print(reviews.shape)
y_train=y_train
reviews=reviews

parameters = {}

# Generate counts from text using a vectorizer.  
# There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = TfidfVectorizer(min_df=4, max_df=0.8, 
                            sublinear_tf=True, use_idf=True)

train_features = vectorizer.fit_transform(reviews)
test_features = vectorizer.transform(test)

with open('vect.pkl','wb') as f:
    pickle.dump(vectorizer,f)

    
# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC(tol=1e-4)
classifier_liblinear.fit(train_features,y_train )
prediction_liblinear = classifier_liblinear.predict(test_features)
# Now we can use the model to predict classifications for our test features.
print(classification_report(y_test, prediction_liblinear))
print("accuracy: {0}".format( accuracy_score(y_test, prediction_liblinear)))


df2=pd.read_csv('data_test1.csv')
test=np.array(df2.loc[:,'text'])
test_features_2=vectorizer.transform(test)
y_pred=classifier_liblinear.predict(test_features)
id=np.array(df2.loc[:,'review_id'])

import csv

with open('laugh.csv', 'w') as f:
    reader = csv.writer(f)
    reader.writerow(['review_id','funny'])
    for i in range(len(y_pred)):
        reader.writerow([id[i],y_pred[i]])

# Compute the error.  
#fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
#print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))

while True:
    sentences = []
    sentence = input("\n\033[93mPlease enter a sentence to get sentiment evaluated. Enter \"exit\" to quit.\033[0m\n")
    if sentence == "exit":
        print("\033[93mexit program ...\033[0m\n")
        break
    else:
        sentences.append(sentence)
        print(sentences)
        input_features = vectorizer.transform(extract_words(sentences))
        prediction = classifier_liblinear.predict(input_features)
        if prediction[0] == 1 :
            print("---- \033[92mpositive\033[0m\n")
        else:
            print("---- \033[91mneagtive\033[0m\n")

import pickle
with  open('funny.pkl','wb') as f:
    pickle.dump(classifier_liblinear,f)            

