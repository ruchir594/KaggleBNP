#!/usr/bin/env python

# Import Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load Training Data
data = pd.read_csv('../data/data/train.csv')

# Extract Float Data
columns = [column for column in data.columns if data[column].dtype == 'float64']
ndata = data[columns]

# Extract Output into Np Array
target = np.array(data['target'])

# Clean NaN Values from Dataset
impute = Imputer(missing_values='NaN', strategy='mean', axis=0)
impute.fit(ndata)
cleaned =impute.transform(ndata)

# Decision Tree Classification
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
scores = cross_val_score(clf, cleaned, target)
print "Decision Tree Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())

# Random Forest Classification
clf=RandomForestClassifier(n_estimators=100)
clf=clf.fit(cleaned,target)
scores = cross_val_score(clf, cleaned, target)
print "Random Forest Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())

# AdaBoost Classifier 
clf=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, cleaned, target)
print "AdaBoost Cross Validation Scores\nMean: %f\nStd: %f" % (scores.mean(), scores.std())
