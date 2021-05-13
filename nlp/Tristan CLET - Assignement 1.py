
"""
# Which method is best for this problem?
# The Multi-Nomial NB have the best results with accurency of 0.98206278 vs for logistic regression it is 0.98026906.
"""

import numpy as np
acc = np.zeros((10,1))

# MNB MODEL

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/Users/tristan/Downloads/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
mnb = MultinomialNB()
y_train=y_train.astype('int')
y_train

mnb.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=mnb.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[0] = count/1115.0

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('/Users/tristan/Downloads/spam.csv',usecols=[0,1],encoding='latin-1')
df.columns=['Status','Message']
print("Length of df",len(df))
print("Number of Spam",len(df[df.Status=='spam']))
print("Number of ham",len(df[df.Status=='ham']))
df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0
df_x=df["Message"]
df_y=df["Status"]
df.head()

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()

cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
cv1.inverse_transform(a[0])

x_testcv=cv1.transform(x_test)
x_testcv.toarray()
lrc = LogisticRegression()
y_train=y_train.astype('int')
y_train

lrc.fit(x_traincv,y_train)
testmessage=x_test.iloc[0]
predictions=lrc.predict(x_testcv)
a=np.array(y_test)

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
acc[1] = count/1115.0
