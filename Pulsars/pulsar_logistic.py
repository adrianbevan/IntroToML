import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

df=pd.read_csv('pulsar_stars.csv')

#Setting x and y and normalize the data
x_data=df.drop(columns='target_class')
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
y=df.target_class.values

compare_score=[]

#training and testing split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

#performing the logistic progression now, we begin with  intializing parameters and creating a sigmoid function
def initialize_weight_bias(dimension):
    w=np.full((dimension,1),0.01)
    #return a new array of given shape and type, filled with fill value
    b=0.
    return w, b

def sigmoid_func(z): #this will be used for the ouput
    y_head=1/(1+np.exp(-z))
    return y_head

#forward and backward propagation
def fw_prop(w, b, x_train, y_train): #feeding forward is a part of back propagation
    #forward
    z=np.dot(w.T, x_train)+b #dot product of the transpose of the weight vector and training inputs
    y_head=sigmoid_func(z)
    y_train=y_train.reshape(14318, 1) 
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log((1-y_head)+1e-15)
    #y_head is yhat in cross-entropy
    cost=(np.sum(loss))/x_train.shape[0] #total loss for the whole model

    #backward
    deri_weight=(np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[0] #deriv of weight wrt no.
    deri_bias=np.sum(y_head-y_train)/x_train.shape[0] #deriv of bias wrt no.
    grad_descents={'deri_weight':deri_weight, 'deri_bias':deri_bias}

    return grad_descents, cost

#updating the parameters (learning)
def update(iter_num, w, b, x_train, y_train, learning_rate):
    cost_list=[]
    index=[]
    for i in range(iter_num+1):
        grad_descents, cost=fw_prop(w, b, x_train, y_train)
        w=w-learning_rate*grad_descents['deri_weight']
        b=b-learning_rate*grad_descents['deri_bias']

        if(i%10==0): #will be used to see the decrease in cost (every ten iterations)
            cost_list.append(cost)
            index.append(i)
            print('Cost after {} iteration = {}'.format(i, cost))

    parameters={'weight':w, 'bias':b}

    return parameters, cost_list, index

#we can now write code that will plot the graph of seeing the decrease in cost
def plot_graph(index, cost_list):
    plt.plot(index, cost_list)
    plt.ylabel('Cost')
    plt.savefig('~/Documents/ML/pulsars/learning_curves/logistic_regression.pdf')
    plt.show()

#predict process
def predict(w, b, x_test):
    #dropping 4 rows in order to be able to properly reshape w
    for x in range(4):
        selection=np.random.randint(0, len(w)-1)
        w=np.delete(w, selection)
    w=w.reshape(3580, 57264)
    z=np.dot(w.T, x_test)+b
    y_head=sigmoid_func(z)
    y_prediction=np.zeros((1, x_test.shape[0]))

    for i in range(y_head.shape[1]):
        if y_head[0, i] <= 0.5:
            y_prediction[0, i]=0
        else:
            y_prediction[0, i]=1

    return y_prediction

#final stretch
def logistic_regression(x_train, y_train, x_test, y_test, iter_num, learning_rate):
    dimension=x_train.shape[0]
    w, b=initialize_weight_bias(dimension)
    parameters, cost_list, index=update(iter_num, w, b, x_train, y_train, learning_rate)

    prediction_test=predict(parameters['weight'], parameters['bias'], x_test)

    #printing errors
    print('Test accuracy: {}%'.format(100-np.mean(np.abs(prediction_test-y_test))*100))

    plot_graph(index, cost_list)

logistic_regression(x_train, y_train, x_test, y_test, iter_num=20, learning_rate=0.5)
