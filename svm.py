#!/usr/bin/env python3
import pandas as pd

titanic = pd.read_csv('Datasets/clean_train.csv')
titanic_test = pd.read_csv('Datasets/clean_test.csv')
del titanic_test['Unnamed: 0']
del titanic['Unnamed: 0']

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(titanic, titanic_test)
from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
  
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(X_train, y_train)

clf = SVC(kernel = 'rbf', C = 1, gamma = 0.000001)
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))


X_predict = data_test.drop(['PassengerId'], axis=1)
clf_predictions = clf.predict(X_predict)
predictions_df = pd.DataFrame(columns = ['Survived'], data = clf_predictions)
predictions_df.to_csv('Predictions/rbf_predictions.csv', index = False)

clf = SVC(kernel = 'linear', C = 1)
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))


X_predict = data_test.drop(['PassengerId'], axis=1)
clf_predictions = clf.predict(X_predict)
predictions_df = pd.DataFrame(columns = ['Survived'], data = clf_predictions)
predictions_df.to_csv('Predictions/linear_predictions.csv', index = False)

