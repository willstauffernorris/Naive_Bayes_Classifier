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


    ## check that there's something in the data (it's not none)




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