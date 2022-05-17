
import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        """
        1.preprocess the raw data
        2.calculate the number of feature 
        3.initalize parameter array 
        """
        (data_processed,
         features_mean,
         features_deviation)=prepare_for_training(data,polynomial_degree,sinusoid_degree,normalize_data)
        
        self.data=data_processed
        self.labels=labels
        self.features_mean=features_mean
        self.features_deviation=features_deviation
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree
        self.normalize_data=normalize_data
        number_features=self.data.shape[1]
        self.theta=np.zeros((number_features,1))
    def train(self,alpha,num_iterations=500):
        cost_history=self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history
        
    def gradient_descent(self,alpha,number_iterations):
        cost_history=[]
        for i in range(number_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history
        
    def cost_function(self,data,labels):
        num_examples=data.shape[0]
        delta=LinearRegression.hypothesis(data,self.theta)-labels
        cost=(1/2)*np.dot(delta.T,delta)/num_examples
        #print(cost.shape)
        #print(cost)
        return cost[0]
        
    def gradient_step(self,alpha):
        number_examples=self.data.shape[0]
        prediction= LinearRegression.hypothesis(self.data,self.theta)
        delta=prediction - self.labels
        theta=self.theta
        theta=theta-alpha*(1/number_examples)*(np.dot(delta.T,self.data)).T
        self.theta=theta
    @staticmethod
    def hypothesis(data,theta):
        prediction=np.dot(data,theta)
        return prediction
    def get_cost(self,data,labels):
        data_processed=prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        return self.cost_function(data_processed,labels)
    def predict(self,data):
        data_processed=prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
        prediction=LinearRegression.hypothesis(data_processed,self.theta)
        return prediction
        
        
        
        