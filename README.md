# Creating a Naive Bayes Classifier from scratch

Read on Medium here:
https://medium.com/@willstauffernorris/creating-a-naive-bayes-classifier-from-scratch-f3df2b398d0e

As a data scientist, it’s fairly simple to bring in a huge number of tools with one very simple line of code. If I want to try running a simple classifier on a dataset, I quickly type:

```
from sklearn.naive_bayes import GaussianNB
```

This is one of the best features of using Python for data science, and taking advantage of the massive open source libraries available. But what’s actually going on under the hood? I set out to write this Naive Bayes Classifier from scratch to better understand it.

The first step was to understand what’s actually going on.

## Understand

The Naive Bayes Classifier is a simple supervised machine learning algorithm that can predict which class a given data point will belong to. It's simplicity means that it's fast, so it works well on large dataset. It also geralizes well (underfits). The "naive" is due to it's assumption of independece of the features in a dataset. This is not usually true in real life, but it simplifies the algorithm and tends to work well in real datasets.

```
P(class|data) = (P(data|class) * P(class)) / P(data)
```
P(class|data) is the probability that the given data point is in a certain class. Eventually, we want to find the class with the highest probability.

P(data|class) is the probabily that the data exists in the given class.

P(class) is the is the prior probability. Regardless of the given data, if that class represents 30% of the data, then this will be 30%.

P(data) is the probability of observing the data. For this classifier, we'll ignore this denominator because it's constant as we compare one data point across different classes.


I understand it better with examples. Say you have symptoms of a rare cancer, and everyone who has this cancer has this symptom. How likely is it that you have the cancer?

```
P(Cancer|Symptoms) = P(Symptoms|Cancer)*P(Cancer)/P(Symptoms)
```

P(Symptoms|Cancer) = 1. The probability that you have symptoms given that you have the rare cancer is 100%

P(Cancer) = 1/1000. This is a rare cancer, so chances are you don't have it.

P(Symptoms) = P(Symptoms|Cancer) * P(Cancer) + P(Symptoms|Non-cancer) * P(Non-cancer)

The first part is the same as above, then we must add the probability that you have the symptoms if you don't have cancer (1/100) multiplied by the probability that you don't have the rare cancer (999/1000)

This adds up to:

```
1 * .001 / 1 * .001 + .01 * .999

== .001 / .01099

== .09 ## Nine percent
```


We will use Bayes Theorem to figure out the probability that a data point belongs to a certain class.
 

## Plan
 
After reading through the conceptual ideas behind the Naive Bayes Classifier, I was ready to dive into the planning process. 

In this 'Plan' phase, I focused on translating the abstract math into pseudocode to form an outline of my algorithm. 

I wrote out all the concepts, and tried to think of the Python implementation. This proved to be trickier than I originally thought, because I was trying not to use any other libraries except NumPy. 

After wrestling with the problem for a while, I also looked up the source code for the scikit learn implementation as well as a few other implementaions. This proved to be extremely helpful, especially for the Gaussian element of the algorithm.


## Execute, part 1

Time to jump into writing real Python code!

There are several parts to this problem:

### 1. Separate the data by class. 
This allows the algorithm to see how the attributes of each class differ.

### 2. Get summary statistics. 
After separating the classes, the algorithm can calculate the mean and standard deviation for each class.

### 3. Calculate summary statistics for each class. 
Now the algorith has mean and standard deviation for each attribute of each class.

### 4. Create a Gaussian probability function. 
This takes in a data point and estimates how likely it is to occur, given a normal distribution of data. This is best explained with an example: 

If the mean and standard deviation are 1, and you're estimating the probability that of a 2 occuring, the probability is .24. A 0 would have the same probability. A 1 has a .39 chance of occuring.

This normal distribution of probability will help in our final step.

### 5. Calculate the probability that a given data point belong in each class. 

Remember Baye's Theorem above? We're going to put it in code.

```
P(class=1|attribute=A, attribute=B) = P(attribute=A|class=1) * P(attribute=B|class=1)* P(class=1)
```

This is slightly different from the first formula, becuase we're comparing the probability that the data belongs to each class to each other class, rather than the raw probability (i.e. we remove the denominator).

This formula also calls out specific attributes. In most datasets, there is more than one attribute - there can be hundreds. We have to calculate the probabilities for each attribute in each class.

## 6. Calculate probilities for each data point in the test set.

This is just doing the same steps above for each data point instead of a single data point. This becomes the .predict method in the class and returns a list of predictions.


## Execute, part 2

Now, let's plug in a dataset into this algorithm.

I started with the very simple (and overused) Iris dataset to test my implementation of the Naive Bayes Classifier against the Scikit learn version.

After preprocessing the data to turn categorical variables into numeric variables, and splitting the dataset into an X_train, X_test, y_train, and y_test pieces, I was ready to test the classifier.

Given the lack of hyper parameters to tune, it’s only a few lines of code to run the classifier.

The scikitlearn version:

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

I instantiate the object, call the .fit and .predict statements, and it returns a list of predicted values.

I set up my Naive Bayes class to take the same inputs as the sklearn version, so the code looks very similar:
wills_gaussian = Gaussian_Naive_Bayes()
wills_gaussian.fit(X_train, y_train)
y_pred = wills_gaussian.predict(X_test)

Depending on the random seed picked in the train/test split, my classifier performed the same or slightly worse on the data. I’m still investigating what the difference is, but it does prove the point that it’s usually better to use an algorithm that has been extensively tested and modified by the open source community, rather than one that was written in a day or two!

## Reflect

After figuring out the basics using the Iris dataset, I tried with a harder datset: classifying stars based on a large number of features. 

https://www.kaggle.com/deepu1109/star-dataset

This dataset has six possible classifications- but only 240 data points, which is a tiny dataset for training. The Naive Bayes classifier still managed to classify most of the test data correctly, with a 85% accuracy.

I haven't tested this dataset on a neural network, but my intuition is that a NN would perform worse, due to the lack of training data.