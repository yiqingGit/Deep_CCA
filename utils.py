"""
	Utils for userful function and activation function
"""
import os
import numpy as np
import random
import scipy
import scipy.linalg as sl


def label2dim(y , dim):
	n = len(y)
	label = np.zeros((n,dim), dtype=np.float32)
	for i in xrange(n):
		d = int(y[i])
		label[i][d] = 1
	return label

def randperm(n , k):
	ret = []
	x = range(n)
	for i in xrange(k):
		t = random.randint(i, n - 1)
		tmp = x[i]
		x[i] = x[t]
		x[t] = tmp
		ret.append(x[i])
	return ret

def compRandomBatchData(data , batch):
	res = []
	for index in xrange(len(batch)):
		res.append(data[batch[index] , :])
	return np.array(res,dtype=np.float32)

def get_grad_activation(activation):
	grad_function = None
	if activation == sigmoid:
		grad_function = grad_sigmoid
	elif activation == tanh:
		grad_function = grad_tanh
	elif  activation == relu:
		grad_function = grad_relu
	elif  activation == None:
		grad_function = None
	else:
		raise ValueError('cann\'t find the gradient activation!')
	return grad_function

def tanh(x):
	return 1.7159 * scipy.tanh(2./3. * x)

def grad_tanh(a):
	return 1.7159 * 2./3. * ( 1. - 1./(1.7159)**2. * a **2. )

def sigmoid(x):
	return 1. / (1. + scipy.exp(-x))

def grad_sigmoid(a):
	return a * (1. - a)

def relu(x):
	return np.maximum(0,x)

def grad_relu(a):
	return np.double(a > 0)
