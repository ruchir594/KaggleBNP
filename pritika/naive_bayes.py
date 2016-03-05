import pandas as pd 
import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
import csv


## Load the Data
#train = pd.DataFrame.from_csv("../data/data/train.csv")
#test = pd.DataFrame.from_csv("../data/data/test.csv")

def naive_bayes(train, test):
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

	y_pred = gnb.fit(train1, target).predict_proba(test1)
	return y_pred


#print(naive_bayes(train, test))


## match along column values


# with open('out.csv', 'w') as fh:
#     writer = csv.writer(fh, delimiter=',')
#     writer.writerow(['ID','PredictedProb'])
#     writer.writerows(enumerate(y_hat))