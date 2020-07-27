from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from bayes import *
import numpy as np
#import pandas as pd

# try not to use pandas
#iris = pd.read_csv("iris.csv")
#print(iris.head())

# csv = np.genfromtxt ('iris.csv', dtype=None, delimiter=',')
# print(csv)




X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# print(X_train)
# print(type(X_train))

# print(y_train)
# print(type(y_train))

# print(y_train[0])





### do a gaussian naive bayes on the iris dataset (SKLEARN)
gnb = GaussianNB()
#y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
#print(y_pred)

print("SKLEARN: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


### do a gaussian naive bayes on the iris dataset(MINE!)

wills_gaussian = Gaussian_Naive_Bayes()

wills_gaussian.fit(X_train, y_train)

y_pred = wills_gaussian.predict(X_test)


print("WILL'S VERSION: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))