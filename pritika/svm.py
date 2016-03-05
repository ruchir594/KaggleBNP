from sklearn import datasets, svm
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import Imputer

#train = pd.DataFrame.from_csv("../data/data/train.csv")
#test = pd.DataFrame.from_csv("../data/data/test.csv")

def bnp_svm(train, test):
	print('bnpsvm')
	## If a value is missing, set it to the average
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

	#print("cleaning data")
	train = train.sample(1000)
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



	#print("training...")
	clf = svm.SVC(gamma=0.001, C=100, probability=True)
	#print("testing")
	clf.fit(train1, target)
	#print("predicting")
	yhat = clf.predict_proba(test1)
	return yhat


#print(bnp_svm(train, test))