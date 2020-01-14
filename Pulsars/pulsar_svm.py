from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikitplot.estimators import plot_learning_curve
import matplotlib.pyplot as plt

df=pd.read_csv('~/Documents/ML/pulsars/pulsar_stars.csv')
x_data=df.drop(columns='target_class')
X=StandardScaler().fit_transform(x_data)
y=df.target_class.values
compare_score=[]

x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

svm=SVC(random_state=42, gamma = 'scale')
svm=svm.fit(x_train, y_train)

svm_score=svm.score(x_test, y_test)*100
compare_score.append(svm_score)

print('Test accuracy: {}%'.format(svm_score))

plot_learning_curve(svm, x_test, y_test)
plt.show()
