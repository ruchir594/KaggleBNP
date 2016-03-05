from sklearn import datasets, svm
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import Imputer
iris = datasets.load_iris()
digits = datasets.load_digits()
train = pd.DataFrame.from_csv("../data/data/train.csv")
test = pd.DataFrame.from_csv("../data/data/test.csv")

#print(classification.describe)
# subset so its just numeric data:
# train1 = train.dropna()
# classification = train1['target']
# train1 = train1.select_dtypes(include=['float64'])
# test1 = test.dropna()
# test1 = test1.select_dtypes(include=['float64'])

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
classification = train['target']
train1 = train.select_dtypes(include=['float64'])
imp.fit(train1)
train1 = imp.transform(train1)


test1 = test.select_dtypes(include=['float64'])
test1 = imp.transform(test1)



print("training...")
clf = svm.SVC(gamma=0.001, C=100)
print("testing")
clf.fit(train1, classification)
print("predicting")
yhat = clf.predict(test1)
print(yhat)
print(len(yhat))