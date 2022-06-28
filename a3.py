#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
titanic = pd.read_csv("Datasets/train.csv")
titanic_test = pd.read_csv("Datasets/test.csv")
def fill_age(dataset,dataset_med):
    for x in range(len(dataset)):
        if dataset["Pclass"][x]==1:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[1,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[1,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[1,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[1,3]["Age"]
        elif dataset["Pclass"][x]==2:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[2,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[2,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[2,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[2,3]["Age"]
        elif dataset["Pclass"][x]==3:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[3,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[3,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[3,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[3,3]["Age"]
            elif dataset["SibSp"][x]==4:
                return dataset_med.loc[3,4]["Age"]
            elif dataset["SibSp"][x]==5:
                return dataset_med.loc[3,5]["Age"]
            elif dataset["SibSp"][x]==8:
                return dataset_med.loc[3]["Age"].median()  
def new_cabin_features(dataset):
    dataset["Cabin A"]=np.where(dataset["Cabin"]=="A",1,0)
    dataset["Cabin B"]=np.where(dataset["Cabin"]=="B",1,0)
    dataset["Cabin C"]=np.where(dataset["Cabin"]=="C",1,0)
    dataset["Cabin D"]=np.where(dataset["Cabin"]=="D",1,0)
    dataset["Cabin E"]=np.where(dataset["Cabin"]=="E",1,0)
    dataset["Cabin F"]=np.where(dataset["Cabin"]=="F",1,0)
    dataset["Cabin G"]=np.where(dataset["Cabin"]=="G",1,0)
    dataset["Cabin T"]=np.where(dataset["Cabin"]=="T",1,0)
def better_ages(df):
    bins = (0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    



titanic["Cabin"]=titanic["Cabin"].fillna("U")
titanic["Cabin"]=titanic["Cabin"].map(lambda x: x[0])
titanic_test["Cabin"]=titanic_test["Cabin"].fillna("U")
titanic_test["Cabin"]=titanic_test["Cabin"].map(lambda x: x[0])
titanic_1=titanic.groupby(["Pclass","SibSp"])
titanic_1_median=titanic_1.median()
titanic["Age"]=titanic["Age"].fillna(fill_age(titanic,titanic_1_median))
titanic_1=titanic_test.groupby(["Pclass","SibSp"])
titanic_1_median=titanic_1.median()
titanic_test["Age"]=titanic_test["Age"].fillna(fill_age(titanic_test,titanic_1_median))
better_ages(titanic)
better_ages(titanic_test)
format_name(titanic)
format_name(titanic_test)
print(titanic.columns)

array, cabin_members = np.unique(titanic["Cabin"], return_counts = True)

plt.xlabel("Cabin"), plt.ylabel("People")
plt.title("Number of people in each cabin ")
plt.bar(array, cabin_members)
plt.savefig("Visualisations/Number of people in each cabin.png")

array, people_in_pclass = np.unique(titanic["Pclass"], return_counts= True)
plt.clf()
plt.xlabel("Pclass"), plt.ylabel("Number of people")
plt.title("Number of people in each class")
plt.bar(array, people_in_pclass)
plt.savefig("Visualisations/Number of people in each class.png")

plt.clf()
plt.xlabel('Age'), plt.ylabel('Surivival rates')
sns.barplot(x="Age", y="Survived", hue="Sex", data=titanic);
plt.savefig("Visualisations/Survival rates of different age groups")

titanic = titanic.drop(['Ticket', 'Name', 'Embarked'], axis=1)
titanic_test = titanic_test.drop(['Ticket', 'Name', 'Embarked'], axis=1)
titanic.to_csv('Datasets/clean_train.csv')
titanic_test.to_csv('Datasets/clean_test.csv')
