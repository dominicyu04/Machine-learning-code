import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("./data/world-happiness-report-2017.csv")
train_data=data.sample(frac=0.8)
test_data=data.drop(train_data.index)
input_param_name="Economy..GDP.per.Capita."
output_param_name="Happiness.Score"

x_train=train_data[[input_param_name]].values
y_train=train_data[[output_param_name]].values
x_train[:3,:]
from sklearn.preprocessing import StandardScaler
x_train_std=StandardScaler().fit_transform(x_train)
x_train_std[:3,:]

print(x_train_std.shape)

def cost_function(X,y,theta):
    return np.sum(np.power(X@theta-y,2))/(2*len(X))

theta=np.zeros((1,1))
print(theta.shape)

cost_init=cost_function(x_train_std,y_train,theta)
print(cost_init)

np.dot(x_train,theta).shape