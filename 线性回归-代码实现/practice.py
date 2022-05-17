import numpy as np
from utils.features import prepare_for_training

class LinearRegression():
    def __init__(self,data,labels,polynominal_degree=0,sinusoid_degree=0,Normalize_data=True):
        (data_processed,
        feature_means,
        feature_deviation)=prepare_for_training(data,polynominal_degree,sinusoid_degree)
        self.data=data_processed
        self.labels=labels
        self.feature_means=feature_means
        self.feature_deviation=feature_deviation
        self.polynominal_degree=polynominal_degree
        self.sinusoid_degree=sinusoid_degree
        self.Normalize_data=Normalize_data
        num_features=self.data.shape[1]
        self.theta=np.zeros((num_features,1))
    
    def train(self,alpha,num_iters=500):
        cost_history=self.gradient_descent(alpha,num_iters)
        return self.theta,cost_history

    def gradient_descent(self,alpha,num_iters):
        cost_history=[]
        for i in range(num_iters):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        num_examples=self.data.shape[0]
        prediction=LinearRegression.hypothesis(self.data,self.theta)
        delta=prediction-self.labels
        theta=self.theta
        #theta=theta-alpha*(1/num_examples)*np.dot(delta.T,self.data).T
        theta =theta-alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta=theta
    def cost_function(self,data,labels):
        num_examples=data.shape[0]
        delta=LinearRegression.hypothesis(data,self.theta)-labels
        #cost=(1/2)*np.square(delta)/num_example
        cost=(1/2)*np.dot(delta.T,delta)/num_examples
        return cost[0][0]


    def hypothesis(data,theta):
        predictions=np.dot(data,theta)
        return predictions

    def get_cost(self,data,labels):
        data_processed=prepare_for_training(data,self.polynominal_degree,self.sinusoid_degree,self.Normalize_data)[0]
        return self.cost_function(data_processed,labels)

    def predict(self,data):
        data_processed=prepare_for_training(data,self.polynominal_degree,self.sinusoid_degree,self.Normalize_data)[0]
        predict=LinearRegression.hypothesis(data_processed,self.theta)
        return predict