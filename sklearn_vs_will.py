from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from class2 import Gaussian_Naive_Bayes
import numpy as np
from csv import reader

#### import csv
dataset = []
with open('iris.csv', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        dataset.append(row)

## Preprocessing

#removing column header
dataset = dataset[1:]

## Convert the strings that got read in to floats
for i in range(len(dataset[0])-1):
    for row in dataset:
        row[i] = float(row[i].strip())


## Converting the string names of the target to int
def string_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

string_to_int(dataset, len(dataset[0])-1)

#create a y target and X features
y = []
for row in dataset:
    y.append(row.pop(-1))

X= dataset

## Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


## COMPARING SKLEARN TO WILL'S ALGORITHM


### do a gaussian naive bayes on the iris dataset (SKLEARN)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("SKLEARN: Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))

### do a gaussian naive bayes on the iris dataset(MINE!)
wills_gaussian = Gaussian_Naive_Bayes()
wills_gaussian.fit(X_train, y_train)
y_pred = wills_gaussian.predict(X_test)

## number incorrect
number_incorrect = 0
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        number_incorrect += 1
## accuracy
accuracy = (len(y_test)-number_incorrect)/len(y_test)*100

print(f'ACCURACY: {round(accuracy,2)}%')
print(f'WILL"S VERSION: OUT OF {len(y_test)} POINTS, {number_incorrect} ARE INCORRECT.')