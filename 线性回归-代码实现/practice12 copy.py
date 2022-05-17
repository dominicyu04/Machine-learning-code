import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  practice11 import LinearRegression

data=pd.read_csv("./data/world-happiness-report-2017.csv")

train_data=data.sample(frac=0.8)
test_data=data.drop(train_data.index)

input_param_name="Economy..GDP.per.Capita."
output_param_name="Happiness.Score"

x_train=train_data[input_param_name].values
y_train=train_data[output_param_name].values

x_test=test_data[input_param_name].values
y_test=test_data[output_param_name].values

plt.scatter(x_train,y_train,label="train_data")
plt.scatter(x_test,y_test,label="test_data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.title("Happy")
plt.show()

num_iterations=500
learning_rate=0.01
linear_regression=LinearRegression(x_train,y_train)
(theta,cost_history)=linear_regression.train(learning_rate,num_iterations)

print("initial lost",cost_history[0])
print("final lost",cost_history[1])