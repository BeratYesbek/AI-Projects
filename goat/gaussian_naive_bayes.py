from statistics import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from random import randint
from numpy import random
from sklearn.pipeline import make_pipeline

data = pd.read_csv("../data/goat.csv")
columns = data.columns.tolist()
columns.remove("2018")


def detect_high_accuracy_with_hill_climbing():
    min = 0
    max = 0
    model = GaussianNB()
    Y = data[["2018"]].values
    Y = np.ravel(Y)
    for i in range(0, 1000):
        random_chosen_column = random.choice(columns, randint(1, 13))
        X = data[random_chosen_column].values
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=True)

        if (len(x_train) != 0 and len(y_train)):
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            acc = accuracy_score(y_test, prediction)
            if (i == 0):
                min = acc
                max = acc
            if (acc > max):
                max = acc
            if (acc < min):
                min = acc
    print("min accuracy: %f" % (min * 100))
    print("max accuracy: %f" % (max * 100))
    print("----------------------------------------")


def detect_accuracy_without_optimization():
    model = GaussianNB()
    x = data[columns].values
    y = data[["2018"]].values
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=True)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    acc = accuracy_score(y_test, prediction)
    print("accuracy without optimization: %f" % (acc * 100))


detect_high_accuracy_with_hill_climbing()
detect_accuracy_without_optimization()
