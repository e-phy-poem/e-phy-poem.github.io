import math
import numpy as np
import pandas as pd

def init_data():
	f = open('C:/联创/机器学习/数据集/diabetes.csv')
	content = pd.read_csv(f)
	df = pd.DataFrame(content)

	f1 = df['preg'].values
	f2 = df['plas'].values
	f3 = df['pres'].values
	f4 = df['skin'].values
	f5 = df['insu'].values
	f6 = df['mass'].values
	f7 = df['pedi'].values
	f8 = df['age'].values
	f9 = df['class'].values
	data = np.array(list(zip(f1, f2, f3, f4, f5, f6, f7, f8, f9)))

	dataMatIn = data[:, 0:-1] 
	classLabels = data[:, -1] 
	dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)
	return dataMatIn, classLabels


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def grad_descent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn) 
	labelMat = np.mat(classLabels).transpose() 
	m, n = np.shape(dataMatrix) 
	weights = np.ones((n, 1))  
	alpha = 0.001 
	maxCycle = 500  

	for i in range(maxCycle): 
		h = sigmoid(dataMatrix * weights) 
		weights = weights + alpha * dataMatrix.transpose() * (labelMat - h) 
	return weights

if __name__ == '__main__':
	dataMatIn, classLabels = init_data()
	r = grad_descent(dataMatIn, classLabels)
	print(r)
