import pandas as pd 
import numpy as np 
from naive_bayes import naive_bayes
from svm import bnp_svm
import operator



## Load the Data
train = pd.DataFrame.from_csv("../data/data/train.csv")
test = pd.DataFrame.from_csv("../data/data/test.csv")
print("naive bayes")
yhat1 = naive_bayes(train, test)
print("svm")
yhat2 = bnp_svm(train, test)



def applyWeights(yhat1, yhat2, weight):
	yhat1 = np.divide(yhat1, weight[0])
	yhat2 = np.divide(yhat2, weight[1])
	pred = np.add(yhat1, yhat2)
	return pred


def getLabel(ar):
	label = []
	softlabel = []
	for x in ar:
		softlabel.append(x[1])
		label.append(round(x[1]))
	return label,softlabel

def getError(pred, true):
	print(pred)
	print(true)
	total = len(pred)
	er = 0.0
	for i in range(len(pred)):
		if pred[i] != true[i][1]:
			er += 1
	return er/float(total)

def cross_val(n, train, weight):
	print("cv")
	err = []
	for i in range(n):
		train_sub = train.sample(10000)
		test_sub = train.sample(1000)
		test_y = test_sub['target']

		y1 = bnp_svm(train_sub, test_sub)
		y2 = naive_bayes(train_sub, test_sub)
		pred = applyWeights(y1, y2, weight)
		lab,s = getLabel(pred)
		err.append(getError(lab, test_y))
	return (sum(err)/float(len(err)))


def selectWeights(train):
	print("sw")
	weights = [[0.1,0.9], [0.2,0.8], [0.3,0.7], [0.4,0.6], [0.5,0.5], [0.6,0.4], [0.7, 0.3], [0.8,0.2], [0.9,0.1]]
	error = []
	for weight in weights:

		error.append(cross_val(3, train, weight))
	min_index, min_value = min(enumerate(error), key=operator.itemgetter(1))
	return(weights[min_index])

print("cross val to select weights")
print(selectWeights(train))
