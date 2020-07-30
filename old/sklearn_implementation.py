from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from bayes import Gaussian_Naive_Bayes
import numpy as np
from csv import reader


#### import csv
dataset = []
with open('iris.csv', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        #if not row:
           # continue
        dataset.append(row)

## This is the head of my dataset
#print(dataset[0:5])


## NEED TO MAKE SURE THE COLUMN NAMES ARE NOT IN IT
#removing column names
dataset = dataset[1:]

#print("-------------")
## Convert the strings that got read in to floats
for i in range(len(dataset[0])-1):
    for row in dataset:
        row[i] = float(row[i].strip())

#print(dataset[0:5])

## Converting the string names of the target to int
# for row in dataset:
#     row[
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
# for row in dataset[:5]:
#     print(row)

#create a y target
y = []
for row in dataset:
    y.append(row.pop(-1))
    #row.remove([-1])


X= dataset
for row in y[:5]:
    print(row)
for row in X[:5]:
    print(row)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# #print(X_train)
#print(type(X_train))

# print(y_train)
# print(type(y_train))
# print(len(y_train))

# print(y_train[0])


# print(len(dataset))
# for row in dataset:
#     print(row)


#X, y = load_iris(return_X_y=True)

for row in X[:5]:
    print(row)
for row in y[:5]:
    print(row)

    


## Train test split
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

print("SKLEARN: Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))


### do a gaussian naive bayes on the iris dataset(MINE!)

wills_gaussian = Gaussian_Naive_Bayes()

wills_gaussian.fit(X_train, y_train)

y_pred = wills_gaussian.predict(X_test)

#print("WILL'S VERSION: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))