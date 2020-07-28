import numpy as np
from math import sqrt, pi, exp

class Gaussian_Naive_Bayes():
    def __init__(self):
        pass

    ## Take in X features and y target and build a model
    def fit(self, X_train, y_train):
        '''
        X is a list of lists. It's the training features.
        y is a list of integers. It's the training target.
        '''
        ## Saving the X train and y train in the object
        self.X_train = X_train
        self.y_train = y_train

        ## Merging the X_train and y_train back together
        ## This is so we can separate by class later
        i = 0
        for row in X_train:
            row.append(y_train[i])
            i += 1

        dataset = X_train

        #call function below
        separated = self.separate_by_class(dataset)
        self.separated = separated

        summarize = self.summarize_by_class(dataset)
        self.summarize = summarize

    
    # Makes a prediction for the y target class, based on the X test 
    def predict(self, X_test):
        '''
        X is a numpy array. It's the test features.
        This function will return the predicted test target.
        '''
        predictions = []

        for row in X_test:
            prediction = self.predict_one(self.summarize, row)
            predictions.append(prediction)
            
        return predictions
    
    ## Separate by class
    def separate_by_class(self, data):
        # creating a dictionary; each key is one of the possible classes
        separated = {}
        for i in range(len(data)):
            row = data[i]
            class_value = row[-1]
            if class_value not in separated:
                separated[class_value] = []
            separated[class_value].append(row)
        #returning a dictionary of each possible value
        return separated


    ## Summarize dataset function
    def summarize_dataset(self, dataset):
        ## calculate mean, std dev, and count for each column in dataset
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

        ## summarize data by class
    def summarize_by_class(self, dataset):
        summaries = {}
        for class_value, rows in self.separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    ## gaussian probability density function
    ## We assume that X value is drawn from a Gaussian distribution (bell curve)
    def calculate_probability(self, x, mean, std):
        exponent = exp(-((x-mean)**2/(2*std**2)))
        return (1/(sqrt(2*pi)*std)) * exponent


        ## class probabilities
    # P(class|data) = P(X|class) * P(class)
    # the probability that it's a certain class given data is the probility that it's X given the class
    #  multiplied by the probability that it's that  class
    # NOTE: the division is removed, because it's redundant when you're comparing across classes

    def calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, std, count = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, std)
        return probabilities

    def predict_one(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            # determine which target class has a higher probability, return this class
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label