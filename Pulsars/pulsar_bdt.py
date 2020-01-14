import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
#Vizualization
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve

data=pd.read_csv('~/Documents/ML/pulsars/pulsar_stars.csv')

#We first decide on how to split the data into features and labels
#features:
features=data.columns[:-1]
X=data[features]
#output:
y=data.target_class

#Now we need to split te data using train_test_split.This requires us to choose said split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#We now make the classifier
classifier=AdaBoostClassifier(DecisionTreeClassifier())

#Training
classifier=classifier.fit(X_train, y_train)
#Preducting the response for the test dataset
y_pred=classifier.predict(X_test)

#Let's see how accurate our model is likely to be
score=round(metrics.accuracy_score(y_test, y_pred)*100, 2)
print('Accuracy = {}%'.format(score))

plot_learning_curve(classifier, X_test, y_test)
plt.show()
