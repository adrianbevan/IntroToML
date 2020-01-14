import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scikitplot.estimators import plot_learning_curve
df=pd.read_csv('~/Documents/ML/pulsars/pulsar_stars.csv')

#Setting x and y and normalize the data
x_data=df.drop(columns='target_class')
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) #scaling
y=df.target_class.values

compare_score=[]

#training and testing split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

lr=LogisticRegression()
lr.fit(x_train, y_train)

lr_score=lr.score(x_test, y_test) * 100
compare_score.append(lr_score)

print('Test accuracy: {}%'.format(lr_score))

plot_learning_curve(lr, x_test, y_test)
plt.show()
