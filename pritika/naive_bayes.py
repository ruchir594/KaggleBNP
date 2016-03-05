import pandas as pd 
import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
import csv


## Load the Data
train = pd.DataFrame.from_csv("../data/data/train.csv")
test = pd.DataFrame.from_csv("../data/data/test.csv")

## If a value is missing, set it to the average
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)


## set up training data
train1 = train.select_dtypes(include=['float64'])
imp.fit(train1)
train1 = imp.transform(train1)
train1 = np.array(train1).astype(float)
## set up real y
target = np.array(train['target']).astype(int)


## set up testing data
test1 = test.select_dtypes(include=['float64'])
test1 = imp.transform(test1)
test1 = np.array(test1).astype(float)


## naive bayes
gnb = GaussianNB()
print(len(test1))
print(len(train1))
y_pred = gnb.fit(train1, target).predict_proba(test1)
m = gnb.fit(train1, target).predict(test1)
print(y_pred)
print(m)
ya = pd.DataFrame(m)
print(ya)

y_hat = []
for x in y_pred:
	y_hat.append(x[1])
#print(y_hat)
#header = ['ID','PredictedProb']
#ya.to_csv("try.csv", header=header)

with open('out.csv', 'w') as fh:
    writer = csv.writer(fh, delimiter=',')
    writer.writerow(['ID','PredictedProb'])
    writer.writerows(enumerate(y_hat))