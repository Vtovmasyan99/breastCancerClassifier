import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values
labelencoder_Y = LabelEncoder()
Y_labeled = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labeled, test_size=0.25, random_state=0)
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)


def doLogisticRegression():
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of Logistic Regression algorithm is: ", accuracy, '%')


def doKNeighborsClassifier():
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of K Nearest Neighbor algorithm is is: ", accuracy, '%')


def doSVC():
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of SVC is: ", accuracy, '%')


def doGaussianNB():
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of Gaussian NB is: ", accuracy, '%')


def doDecisionTree():
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of Decision Tree Classifier is: ", accuracy, '%')


def doRandomForestClassifier():
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) * 100
    print("The accuracy of Random Forest Classifier  is: ", accuracy, '%')


def main():
    doLogisticRegression()
    doKNeighborsClassifier()
    doSVC()
    doGaussianNB()
    doDecisionTree()
    doRandomForestClassifier()


if __name__ == '__main__':
    main()
