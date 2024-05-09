# code
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv")
print(data.head())

print(data.describe())
print(data.isnull().sum())

plt.figure(figsize=(15, 10))
plt.title("Product Purchased By People Through Social Media Marketing")
sns.histplot(x="Age", hue="Purchased", data=data)
plt.show()

#The preceding visualization illustrates the target audience's over-45 demographic is more interested in 
buying the goods.

plt.title("Product Purchased By People According to Their Income")
sns.histplot(x="EstimatedSalary", hue="Purchased", data=data)
plt.show()

The visualization indicates that individuals in the target group who earn more than $90,000 per month are more 
likely to buy the goods.

x = np.array(data[["Age", "EstimatedSalary"]])
y = np.array(data[["Purchased"]])

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(xtrain, ytrain)
dt_predictions = dt_model.predict(xtest)
print("Decision Tree Classifier:")
print(classification_report(ytest, dt_predictions))

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(xtrain, ytrain)
rf_predictions = rf_model.predict(xtest)
print("Random Forest Classifier:")
print(classification_report(ytest, rf_predictions))

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(xtrain, ytrain)
nb_predictions = nb_model.predict(xtest)
print("Naive Bayes Classifier:")
print(classification_report(ytest, nb_predictions))

# Logistic Regression Classifier
lr_model = LogisticRegression()
lr_model.fit(xtrain, ytrain)
lr_predictions = lr_model.predict(xtest)
print("Logistic Regression Classifier:")
print(classification_report(ytest, lr_predictions))

# Support Vector Machine (SVM) Classifier
svm_model = SVC()
svm_model.fit(xtrain, ytrain)
svm_predictions = svm_model.predict(xtest)
print("Support Vector Machine (SVM) Classifier:")
print(classification_report(ytest, svm_predictions))
