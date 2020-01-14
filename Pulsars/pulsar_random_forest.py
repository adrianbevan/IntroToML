import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scikitplot.estimators import plot_learning_curve

compare_score=[]

df=pd.read_csv('~/Documents/ML/pulsars/pulsar_stars.csv')

x_data=df.drop(columns='target_class')
X=StandardScaler().fit_transform(x_data)
y=df.target_class.values

x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

rf=RandomForestClassifier(random_state=42, n_estimators=10)

rf.fit(x_train, y_train)

rf_score=rf.score(x_test, y_test)*100
compare_score.append(rf_score)

print('Test accuracy: {}%'.format(rf_score))

plot_learning_curve(rf, x_test, y_test)
plt.show()
