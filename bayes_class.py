from sklearn_vs_will import X_train, X_test, y_train, y_test
import numpy as np
from math import sqrt, pi, exp

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
    
    ## Makes a prediction for the y target class, based on the X test 
    def predict(self, X):
        '''
        X is a numpy array. It's the test features.
        This function will return the predicted test target.
        '''
        #return pred_target
        pass



## Merging the X_train and y_train back together
i = 0
for row in X_train:
   row.append(y_train[i])
   i += 1

dataset = X_train


# 1) Calculate prior probablilites **for each** class label

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
    #returning a dictionary of each possible value
    return separated

separated = separate_by_class(dataset)
# for label in separated:
#     print(label)
#     for row in separated[label]:
#         print(row)


## Summarize dataset function
def summarize_dataset(dataset):
    ## calculate mean, std dev, and count for each column in dataset
    summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
    #print("---------")
    #print(summaries[-1])

# summary = summarize_dataset(dataset)
# print(summary)

#print(len(dataset))


## summarize data by class
def summarize_by_class(dataset):
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

summaries = summarize_by_class(dataset)
# for label in summaries:
#     print(label)
#     for row in summaries[label]:
#         print(row)



## gaussian probability density function
## We assume that X value is drawn from a Gaussian distribution (bell curve)
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
        # determine which target class has a higher probability, return this class
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# prediction = predict(summaries, dataset[16])
# print(prediction)
# print(dataset[16])




summarize = summarize_by_class(dataset)
#print(summarize)

predictions = []

# prediction = predict(summarize, X_test[0])
# print(X_test[0])
# print(y_test[0])
# print(prediction)

for row in X_test:
    prediction = predict(summarize, row)
    predictions.append(prediction)


# print(predictions)
# print(y_test)

## number incorrect
number_incorrect = 0
for i in range(len(y_test)):
    if y_test[i] != predictions[i]:
        number_incorrect += 1

accuracy = (len(y_test)-number_incorrect)/len(y_test)*100

print(f'ACCURACY: {round(accuracy,2)}%')
print(f'OUT OF {len(y_test)} POINTS, {number_incorrect} ARE INCORRECT.')



# def naive_bayes(train, test):
#     summarize = summarize_by_class(train)
#     predictions = []
#     for row in test:
#         print(row)
#         output = predict(summarize, row)
#         predictions.append(output)
#     return predictions

# naive_bayes(dataset, y_train)


## put it all together in a more readable way
## put it all into a class!!