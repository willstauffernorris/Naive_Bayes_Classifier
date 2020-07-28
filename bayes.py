'''
Your algorithm, implemented as a Python class, should have the following methods: fit and predict. 
You are only allowed to use base Python, numpy, and scipy for the implementation of the core algorithm of your choice. 
(For visualization and analysis, you can use other libraries.) You may reference any outside materials that you need, but copying and pasting another open-source implementation is strictly prohibited.

You'll then use your implementation on an appropriate set of data and compare the results that you got from your implementation against the results yielded by the versions in the sklearn library.
'''




## Two important things to figure out:

# 1) what format is the data I pass in going to be

#numpy array



# 2) what is the logic behind bayes

#don't conflate these things!


class Gaussian_Naive_Bayes():
    def __init__(self, priors=None):
        # this is the prior probability- crux of Bayes theorem
        self.priors = priors

        #also need variable smoothing, but I'm not sure what that is yet
        #self.variable_smoothing
    
    ## Take in X features and y target and build a model
    def fit(self, X, y):
        '''
        X is a numpy array. It's the training features.
        y is a numpy array. It's the training target.
        '''
        self.X = X
        self.y = y



    # 1) Calculate prior probablilites **for each** class label

       # 2) Calculate conditional probability with each attribute
        
        #probability that each feature is the way it is
        # probability that the class is 1, 2, or 3
        #self.class0 = how many times that class appears/how many total observations

    # Calculate posterior probabilities
        # prob that it's a certain class given the features

    # put them together
        # prob target * posterior prob / prob feature 

    
    #do this for each of the potential target classes

#--------

    # determine which target class has a higher probability, return this class
    #  




    
    ## Makes a prediction for the y target class, based on the X test 
    def predict(self, X):
        '''
        X is a numpy array. It's the test features.
        This function will return the predicted test target.
        '''

        #return pred_target
        pass



    ######################
    ## This is taking a subset of the iris datset










    ## this is where I'm going to implement the example from Wikipedia.

    # # Training set:
    # ## The data is "Gender", "Height (ft)", "weight (lbs)", "foot size (inches)"
    # X_train = [
    # ['male', 6, 180, 12],
    # ['male', 5.92, 190, 11],
    # ## another two lines of male
    # ['female', 5, 100, 6],
    # ['female', 5.5, 150, 8]
    # ## another two lines of female
    # ]

    # X_test = ['male', 'male', 'female', 'female']


from csv import reader

#### import iris
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
#print("-------------")
#print(dataset[:5])


## Train test split
# from sklearn.model_selection import train_test_split



## Separate by class
def separate_by_class(data):
    # creating a dictionary; each key is one of the possible classes
    separated = {}
    for i in range(len(data)):
        row = dataset[i]
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row)
    return separated


separated = separate_by_class(dataset)
# print("-------------")
# for label in separated:
#     print(label)
#     for row in separated[label]:
#         print(row)



## Summarize dataset
import numpy as np

## get mean

## calculate std deviation
## calculate mean, std dev, and count for each column in dataset

### SUMMARIZE
def summarize_dataset(dataset):
    summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
    #print("---------")
    #print(summaries[-1])

#summary = summarize_dataset(dataset)
#print(summary)

#print(len(dataset))



## summarize data by class
def summarize_by_class(dataset):
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# for label in summaries:
#     print(label)
#     for row in summaries[label]:
#         print(row)





## gaussian probability density function
## We assume that X value is drawn from a Gaussian distribution (bell curve)

from math import sqrt, pi, exp

def calculate_probability(x, mean, std):
    exponent = exp(-((x-mean)**2/(2*std**2)))
    return (1/(sqrt(2*pi)*std)) * exponent

## This is an example bell curve with mean of 1, std of 1.
## The likelihood that the x is 2 or 0 is .24
#print(calculate_probability(2.0, 1.0, 1.0))





## class probabilities
# P(class|data) = P(X|class) * P(class)
# the probability that it's a certain class given data is the probility that it's X given the class
#  multiplied by the probability that it's that  class
# NOTE: the division is removed, because it's redundant when you're comparing across classes


def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, std, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, std)
    return probabilities

#probabilities = calculate_class_probabilities(summaries, dataset[0])

#print(dataset[0])
#print(probabilities)


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# prediction = predict(summaries, dataset[0])
# print(prediction)
# print(dataset[0])


## put it all together in a more readable way
## put it all into a class!!


# from sklearn.model_selection import train_test_split


# #print(type(dataset))

# #create a y target
# y = []
# for row in dataset:
#     y.append([row.pop(-1)])
#     #row.remove([-1])

# X= dataset
# print(y)
# #print(X)

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



def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = []
    for row in test:
        print(row)
        output = predict(summarize, row)
        predictions.append(output)
    return predictions



#naive_bayes(X, y)