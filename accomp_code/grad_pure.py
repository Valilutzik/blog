import  pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification



class_data = make_classification(n_samples = 5000, n_features = 2, n_redundant = 0, class_sep = 2.5, n_clusters_per_class = 1, random_state = 32)data= class_data[0]
target = class_data[1]
#add intercept
data = np.append(np.ones((data.shape[0],1)),data,axis=1)

def sigmoid(X):	
	return 1 /  (1+np.exp(-X))

def  gradDescent(data,target): 
	learning_rate = 0.01
	num_iter = 1000
	m,n = data.shape
	weights = np.ones((n,1))
	for i in xrange(num_iter):
		f = sigmoid(np.dot(data,weights))
		error = (target.reshape(-1,1) - f)
		weights = weights + learning_rate *  (np.dot(data.T,error)) 
	return weights	

def  plot_boundary(weights, data, target):  
	colors = ["red","blue"]
	for i in [0,1]:
		x1 = data[:,1][target == i]
		x2 = data[:,2][target == i]
		plt.scatter(x1,x2, c=colors[i])
	plt.legend(["class1","class2"])	
	plt.xlim(min(data[:,1]) - 1, max(data[:,1]) + 1)
	plt.ylim(min(data[:,2]) -1 ,max(data[:,2]) + 1)
	x1 = np.arange(min(data[:,1]),max(data[:,1]),0.01)
	x2 = (-weights[0] - weights[1]*x1)/weights[2]
	plt.plot(x1,x2)
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	plt.legend()
	plt.title("Logistic Regression Separation")
	plt.show()		

def  plot_1Dsigmoid(weights, data, target):
	#Weights at step = 0
	w_0 = np.ones((data.shape[1],1))
	plt.plot(np.sort(data[:, 2]), np.sort(sigmoid(np.dot(data,w_0)),axis=0), "ro-", label="Before Optimization")

	#plt.plot(np.sort(data[:, 2]), np.sort(sigmoid(np.dot(data,weights)),axis=0), "bv-", label = "After Optimization") 	
	plt.plot(np.sort(data[:, 2]), np.sort(sigmoid(np.dot(data,clf.coef_[0])),axis=0), "bv-", label = "After Optimization") 	

	plt.scatter(np.sort(data[:, 2]),target,c=target)
	plt.legend(loc=4)
	plt.xlabel("$x_1$")
	plt.ylabel("$y$")
	plt.grid(True)
	plt.show()

def contour_plot_error_function?

#########Classification with scikit###########
import zipfile
import StringIO
import requests

r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip")
z = zipfile.ZipFile(StringIO.StringIO(r.content))
df = pd.read_csv(z.open("bank.csv"), delimiter=";")

def data_processing(df):
	data = df._get_numeric_data().values
	data = np.append(np.ones((data.shape[0],1)),data,axis=1)
	target = pd.Categorical(target).codes
	return data, target

