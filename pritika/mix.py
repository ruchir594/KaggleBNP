import pandas as pd 
import numpy as np 
from naive_bayes import naive_bayes
from svm import bnp_svm



## Load the Data
train = pd.DataFrame.from_csv("../data/data/train.csv")
test = pd.DataFrame.from_csv("../data/data/test.csv")
print("naive bayes")
yhat1 = naive_bayes(train, test)
print("svm")
yhat2 = bnp_svm(train, test)

print(yhat1)
print(yhat2)



weight = [2, 2]
yhat1 = np.divide(yhat1, weight[0])
yhat2 = np.divide(yhat2, weight[1])

	
pred = np.add(yhat1, yhat2)
print(pred)



def getLabel(ar):
	label = []
	softlabel = []
	for x in ar:
		softlabel.append(x[1])
		label.append(round(x[1]))
	return label,softlabel

lab,soft = getLabel(pred)
print(sum(lab))
print(len(lab))

