#README file

#Sentiment Analysis on U.S. Airline Dataset

Date: 12/05/2018
#Authors:
Meena Rapaka - Role: (Data Cleaning and PreProcesing)
Siva Naga Lakshmi Karamsetty - Role: (Modelling)
Ying ke - Role: ( Data Visualization )


Scope:

Using sentiment analysis, we apply different models in order to recognize the best performing model.

1. Introduction:
The project is regarding analysis about the problems of each major U.S airline such as American Airlines,
Delta, Southwest, United, US Airways, Virgin America. Twitter data was scraped from the airlines and
is categorized into positive, negative and neutral tweets, followed by categorizing negative reasons
further such as “delay” or “rude service”. 14640 tweets from 7700 users were analyzed as a part of it.
The dataset is processed, and modelling techniques are applied further to get desired results.
Different NLP techniques and machine learning models are used to address the problem defined.

2. Outline of solution:
Natural language processing techniques such as word clouds, TF-IDF, Bag of words, ngrams, sentiment analysis etc.,
are used to process the data. Also, machine learning techniques such as logistic regression, random forest,
support vector machine, K-Nearest Neighbor, Decision tree, Naïve Bayes are applied to predict
the outcome variables. A baseline model, Support vector machine classifier is performed to
check the accuracy and use it as a baseline for rest our analysis. Then, we compute the accuracies
for various models to recognize the best performing model among the different models we applied.
We got the best accuracy for sentiment analysis with Logistic regression with an accuracy of 77% for
both TF-IDF and Bag of Words model compared to the baseline accuracy of 64.5%.


3. Examples of program input and output:
Following the steps mentioned in the project_msy.py python file would serve as input to the 
program. And output of the program is the results of each model such as mentioned below:

Baseline model (Support Vector Machine - TF-IDF):64.5%
Baseline model (Support Vector Machine - Bag of Words):64.5%
Logistic Regression with TF-IDF:77%
Logistic Regression with Bag of Words:77.3%
Random Forest with TF-IDF:75.2%
Random Forest with Bag of Words:73%
Naïve Bayes with TF-IDF:75%
Naive Bayes with Bag of Words:73.5%
Decision Trees with TF-IDF:71.1%
Decision Trees with Bag of Words:67.9%
K Nearest Neighbor with TF-IDF:40.4%
K Nearest Neighbor with Bag of Words:60.6%



