# Creating a Naive Bayes Classifier from scratch

As a data scientist, it’s fairly simple to bring in a huge number of tools with one very simple line of code. If I want to try running a simple classifier on a dataset, I quickly type:

```
from sklearn.naive_bayes import GaussianNB
```

This is one of the best features of using Python for data science, and taking advantage of the massive open source libraries available. But what’s actually going on under the hood? I set out to write this Naive Bayes Classifier from scratch to better understand it.

The first step was to understand what’s actually going on.

## Understand

(put this in Latex)
```
P(class|data) = (P(data|class) * P(class)) / P(data)
```

What's the conceptual framework in the Naive Bayes Classifier?

## Plan

Look through resources like wikipedia, etc
write a lot of comments/pseudocode for an outline
find a few code implementations online to help with the trickiest parts


## Execute
I started with the very simple (and overused) Iris dataset to test my implementation of the Naive Bayes Classifier against the Scikit learn version.

After preprocessing the data to turn categorical variables into numeric variables, and splitting the dataset into an X_train, X_test, y_train, and y_test pieces, I was ready to test the classifier.

Given the lack of hyper parameters to tune, it’s only a few lines of code to run the classifier.

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
