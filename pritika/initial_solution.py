# Read in sample solution
import numpy
import pandas as pd
import random

random_sol = numpy.loadtxt(open("../data/data/sample_submission.csv"), delimiter=',', skiprows = 1)
df = pd.DataFrame(random_sol)
df[0] = df[0].astype(int)


random_prediction = []
for i in range(len(df.index)):
	random_prediction.append(random.uniform(0,2))

df[1] = df[1] * random_prediction


headers = ["ID", "PredictedProb"]
df.to_csv("random_sol.csv", header = headers, index=False)
