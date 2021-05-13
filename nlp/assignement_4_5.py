#######################################################
# CLET Tristan - e201475
# ---------------------------------
# Natural Language Processing
# Final Assignement
#######################################################

# Maths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Word checkers
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, plot_roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier

# Global Variables setup
porter = PorterStemmer()
nltk.download('stopwords')
stop_words = stopwords.words('french')
vc_tf_idf = TfidfVectorizer()

# Sample configuration
TRAIN_ROWS = 1000
TEST_ROWS = 300

# Function to summarize the number of keywords
def cleaningKeywords(keywords):
    wordcounts=keywords.split(";")
    text=""
    i = 0
    for words in wordcounts:
        countWords=words.split(":")
        i = i+1
        # segmentisation by counting format to words
        if len(countWords)>1 and not countWords[0] in stop_words :
            countWords[0]=porter.stem(countWords[0])
            text+=(countWords[0]+" ")*int(countWords[1])
    return text


# Clean train data
train_clean=pd.read_csv("/Users/tristan/Desktop/nlp/train.csv")
train_clean.dropna(inplace=True)
train_clean["sex"]=train_clean["sex"].replace("M",1)
train_clean["sex"]=train_clean["sex"].replace("F",0)
train_clean=train_clean[train_clean["keywords"].str.contains(":")][0:TRAIN_ROWS]
train_clean["keywords"]=train_clean["keywords"].map(lambda x: cleaningKeywords(x))
train_clean.dropna(inplace=True)

# Clean test data
test_clean=pd.read_csv("/Users/tristan/Desktop/nlp/test.csv")
test_clean.dropna(subset=["keywords"], inplace=True)
test_clean=test_clean[test_clean["keywords"].str.contains(":")][0:TEST_ROWS]
test_clean["keywords"]=test_clean["keywords"].map(lambda x: cleaningKeywords(x))
test_clean.dropna(subset=["keywords"], inplace=True) # cleaning again if no keyword is attributed

# Apply TfidfVectorizer
X_train,X_test,y_train,y_test,z_train,z_test = train_test_split(train_clean["keywords"],train_clean["sex"],train_clean["age"], test_size = 0.2, random_state = 42)
vc_tf_idf.fit(X_train.apply(lambda x: np.str_(x)))
X_train_tf = vc_tf_idf.transform(X_train.apply(lambda x: np.str_(x)))
X_train_tf[:3].nonzero()


####
# Applying Mltinomial, logistic regression and random forest for sex
####
X_test_tf = vc_tf_idf.transform(X_test.apply(lambda x: np.str_(x)))
predictionsex = dict()

logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_tf,y_train)
predictionsex["logreg"] = logreg.predict(X_test_tf)
ax = plt.gca()
test_rf = plot_roc_curve(logreg, X_test_tf, y_test,ax=ax, alpha=0.8, name="test")
train_rf = plot_roc_curve(logreg,X_train_tf, y_train,ax=ax, alpha=0.8, name="train")
print("Logistic Regression accuracy : "+str(np.round(accuracy_score(y_test,predictionsex["logreg"])*100,3))+"%") 
print(classification_report(y_test,predictionsex["logreg"])) #Test result on LogisticRegression
plt.show() 

mnb = MultinomialNB()
mnb.fit(X_train_tf,y_train)
predictionsex["mnb"] = mnb.predict(X_test_tf)
ax = plt.gca()
test_rf = plot_roc_curve(mnb, X_test_tf, y_test,ax=ax, alpha=0.8, name="test")
train_rf = plot_roc_curve(mnb,X_train_tf, y_train,ax=ax, alpha=0.8, name="train")
print("MultinomialNB accuracy : "+str(np.round(accuracy_score(y_test,predictionsex["mnb"])*100,3))+"%") 
print(classification_report(y_test,predictionsex["mnb"])) #Test result on MultinomialNB
plt.show() 

rfc= RandomForestClassifier(n_estimators=2000,max_depth=8)
rfc.fit(X_train_tf,y_train)
predictionsex["rfc"] = rfc.predict(X_test_tf)
ax = plt.gca()
test_rf = plot_roc_curve(rfc, X_test_tf, y_test,ax=ax, alpha=0.8, name="test")
train_rf = plot_roc_curve(rfc,X_train_tf, y_train,ax=ax, alpha=0.8, name="train")
print("Random Forst accuracy : "+str(np.round(accuracy_score(y_test,predictionsex["rfc"])*100,3))+"%") 
print(classification_report(y_test,predictionsex["rfc"])) #Test result on Randomforest Classifier
plt.show() 

X_test_final = vc_tf_idf.transform(test_clean["keywords"].apply(lambda x: np.str_(x)))
test_result=pd.DataFrame()
test_result["ID"]=test_clean["ID"]
test_result["sex_pred"]=rfc.predict(X_test_final) # Random Forest has the best accuracy
test_result["sex_pred"]=test_result["sex_pred"].replace(1, "M")
test_result["sex_pred"]=test_result["sex_pred"].replace(0, "F")

####
# Applying SGD regressor for age
####
sgd = SGDRegressor(alpha=0.00001,max_iter=10000,random_state=None)
sgd.fit(X_train_tf,z_train)
test_result["age_pred"]=np.rint(sgd.predict(X_test_final)).astype(np.int64)
test_result.to_csv("/Users/tristan/Desktop/nlp/test_result.csv",index=False,sep=';')