"""
	Utils for cost function 
"""
import os
import numpy as np
import random
import copy
import scipy
import scipy.linalg as sl

from myIO import *
from utils import *

def euclid_cost(x , y):
	n = x.shape[0]
	error_ = y - x
	cost = 0.5 *  np.sum(error_ **2. ) / n	
	grad = np.array( - error_ / n , dtype= np.float32)
	return cost , grad

def softmax(x , y):
	n = x.shape[0]
	max_x =  np.max(x , axis = 1)
	max_matrix = np.tile(max_x , (x.shape[1] , 1)).T
	x = x - max_matrix
	ex  = np.exp(x, dtype=np.float32)

	sum_ex = np.sum(ex , axis = 1)
	sum_matrix = np.tile(sum_ex , (ex.shape[1] , 1)).T
	prob = ex / sum_matrix
	#print prob	
	err = (y - prob) 

	log_prob = np.log(prob, dtype=np.float32)
	cost = - np.sum(np.multiply(y ,log_prob )) / n	
	grad  = - err / n
	return cost, grad

def euclid(x1 , x2):
	n = x1.shape[0]
	error_ = x2 - x1
	cost = 0.5 *  np.sum(error_ **2. ) / n	
	grad_x1 = np.array( - error_ / n , dtype= np.float32)
	grad_x2 = np.array( error_ / n , dtype= np.float32)
	return cost , grad_x1, grad_x2

def cosin_cost(x1 , x2):	
	# x1 is a matrix n x r
	# x2 is a matrix n x r
	# cost is a similarity between x1 and x2
	# Note: For the vector t that have || t || = (t1^2 + t2^2 + ... + tr^2)

	n = x1.shape[0]
	#####################################################################
	# 1 / ||x1||^1/2
	square_x = x1 * x1	
	norm_factor = np.sum(square_x , axis=1)
	len_x1 = np.sqrt(norm_factor)	
	dlen_x1 = np.divide(1. , len_x1)
	# 1 / ||x2||^1/2
	square_x = x2 * x2
	norm_factor = np.sum(square_x , axis=1)
	len_x2 = np.sqrt(norm_factor)	
	dlen_x2 = np.divide(1. , len_x2)

	###################################################################
	# x1 / ||x1||^1/2 and x2 / ||x2||^1/2	
	norm_x1 = x1 * np.tile(dlen_x1 , (x1.shape[1] , 1)).T
	norm_x2 = x2 * np.tile(dlen_x2 , (x2.shape[1] , 1)).T 
	#sim_x12 = x1 * x2 / (||x1||^1/2 * ||x2||^1/2)
	sim_x12 = (norm_x1 * norm_x2).sum(axis=1)
	dlen_x12 = dlen_x1 * dlen_x2
	###################################################################
	
	# derivated_x1 : First x2 / (||x1||^1/2 * ||x2||^1/2)
	d1_x1  =  x2 * np.tile(dlen_x12,(x1.shape[1] , 1)).T
	# (sim_x12 / ||x1||) * x1 
	square_factor = np.tile(sim_x12 * (dlen_x1 **2) , (x1.shape[1] , 1)).T
	d2_x1 = square_factor * x1 
	# derivated_x1 : Then x2 / (||x1||^1/2 * ||x2||^1/2) -  sim_x12 * x1 / ||x1||
	derivated_x1 = d1_x1 - d2_x1

	# derivated_x2 : First x1 / (||x1||^1/2 * ||x2||^1/2)
	d1_x2  =  x1 * np.tile(dlen_x12 ,(x2.shape[1] , 1)).T
	# (sim_x12 / ||x2||) * x2
	square_factor = np.tile(sim_x12 * (dlen_x2 **2)  , (x2.shape[1] , 1)).T
	d2_x2 = square_factor * x2 
	# derivated_x2 : Then  x1 / (||x1||^1/2 * ||x2||^1/2) - sim_x12 * x2 / ||x2|| 
	derivated_x2 =  d1_x2 - d2_x2
	####################################################################	
	err = 1 - sim_x12
	cost = np.sum(err) / n
	grad_x1 = - derivated_x1 / n
	grad_x2 = - derivated_x2 / n
	return cost, grad_x1, grad_x2

def cm_loss(x1,x2,y):	
	n = x1.shape[0]
	square_x = x1 * x1	
	norm_factor = np.sum(square_x , axis=1)
	len_x1 = np.sqrt(norm_factor)	
	dlen_x1 = np.divide(1. , len_x1)
	norm_x1 = x1 * np.tile(dlen_x1 , (x1.shape[1] , 1)).T
	square_x = x2 * x2
	norm_factor = np.sum(square_x , axis=1)
	len_x2 = np.sqrt(norm_factor)	
	dlen_x2 = np.divide(1. , len_x2)
	norm_x2 = x2 * np.tile(dlen_x2 , (x2.shape[1] , 1)).T 

	score_layer = np.dot(norm_x1,norm_x2.T)
	gt_layer = np.dot(y,y.T)
	#print score_layer,gt_layer

	cost,grad = euclid_cost(score_layer, gt_layer)

	grad_x1 = np.dot(grad, norm_x2)
	grad_x2 = np.dot(grad.T, norm_x1)

	derivate_x1 = np.tile(1. * dlen_x1 , (x1.shape[1] , 1)).T - norm_x1 * np.tile(dlen_x1, (x1.shape[1] , 1)).T
	derivate_x2 = np.tile(1. * dlen_x2 , (x2.shape[1] , 1)).T - norm_x2 * np.tile(dlen_x2 , (x2.shape[1] , 1)).T
	grad_x1 = grad_x1 * derivate_x1
	grad_x2 = grad_x2 * derivate_x2 
	return cost, grad_x1, grad_x2

def grad_norm(x):
	I = np.eys(x.shape[1])
	xx = np.dot(x.T, x)
	norm = np.sqrt((x*x).sum())
	return I/norm - xx/norm**3

def norm_function(x, y):
	n = x.shape[0]
	xx = x * x
	I = np.eye(x.shape[1])
	err = y - norm_x
	cost = 0.5 * np.sum(err **2) / n
	grad = - err * grad / n

	return cost, grad


def check_cm_gradient(cost_function = cm_loss,epsilon = 1e-2,er = 1e-3):
	x = np.random.rand(2,3)
	y = np.random.rand(2,3)	
	z = np.eye(2)
	cost, grad_x,grad_y = cost_function(x,y,z)
	for i in xrange(2):
		for j in xrange(3):
			x_p = copy.deepcopy(x)
			x_p[i][j] += epsilon
			cost_p, grad,tmp= cost_function(x_p,y,z)
			
			x_m = copy.deepcopy(x)
			x_m[i][j] -= epsilon
			cost_m, grad,tmp= cost_function(x_m,y,z)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad_x[i][j])
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad_x[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')

	for i in xrange(2):
		for j in xrange(3):
			y_p = copy.deepcopy(y)
			y_p[i][j] += epsilon
			cost_p, grad,tmp = cost_function(x,y_p,z)
			
			y_m = copy.deepcopy(y)
			y_m[i][j] -= epsilon
			cost_m, grad,tmp = cost_function(x,y_m,z)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad_y[i][j])
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad_y[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')

def check_cosin_gradient(cost_function = cosin_cost,epsilon = 1e-2,er = 1e-3):
	x = np.random.rand(2,3)
	y = np.random.rand(2,3)	
	cost, tmp, grad = cost_function(x,y)
	for i in xrange(2):
		for j in xrange(3):
			y_p = copy.deepcopy(y)
			y_p[i][j] += epsilon
			cost_p, tmp, grad_p = cost_function(x,y_p)

			y_m = copy.deepcopy(y)
			y_m[i][j] -= epsilon
			cost_m,tmp,grad_m = cost_function(x,y_m)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad[i][j])
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')

def check_euclid_gradient(cost_function = euclid_cost,epsilon = 1e-2,er = 1e-3):
	x = np.random.rand(2,3)
	y = np.random.rand(2,3)	
	z = np.eye(2)
	score = np.dot(x,y.T)
	cost, grad = euclid_cost(score,z)
	grad_x = np.dot(grad,y)
	grad_y = np.dot(grad.T,x)
	for i in xrange(2):
		for j in xrange(3):
			x_p = copy.deepcopy(x)
			x_p[i][j] += epsilon
			score_xp = np.dot(x_p,y.T)
			cost_p, grad = euclid_cost(score_xp,z)
			
			x_m = copy.deepcopy(x)
			x_m[i][j] -= epsilon
			score_xm = np.dot(x_m,y.T)
			cost_m, grad = euclid_cost(score_xm,z)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad_x[i][j])
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad_x[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')

	for i in xrange(2):
		for j in xrange(3):
			y_p = copy.deepcopy(y)
			y_p[i][j] += epsilon
			score_xp = np.dot(x,y_p.T)
			cost_p, grad = euclid_cost(score_xp,z)
			
			y_m = copy.deepcopy(y)
			y_m[i][j] -= epsilon
			score_xm = np.dot(x,y_m.T)
			cost_m, grad = euclid_cost(score_xm,z)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad_y[i][j])
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad_y[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')

def check_function_gradient(cost_function = softmax,epsilon = 1e-2,er = 1e-3):
	x = np.random.rand(2,3)
	y = np.random.rand(2,3)	
	## Softmax Special Process
	if(cost_function == softmax):
		max_y =  np.max(y , axis = 1)
		tile_y = np.tile(max_y , (y.shape[1] , 1)).T
		y=np.double(y>=tile_y)
	##
	cost, grad = cost_function(x,y)
	for i in xrange(2):
		for j in xrange(3):
			x_p = copy.deepcopy(x)
			x_p[i][j] += epsilon
			cost_p, grad_p = cost_function(x_p,y)

			x_m = copy.deepcopy(x)
			x_m[i][j] -= epsilon
			cost_m, grad_m = cost_function(x_m,y)

			grad_pm = (cost_p - cost_m) / (2 * epsilon)
			e = np.abs(grad_pm - grad[i][j])
		
			print('e : {} \t grad_pm : {}  -- grad_W: {}'.format(e,grad_pm,grad[i][j]))
			if(e > er):
				print('numerical gradient checking failed !' )
				break
			else:
				print('numerical gradient checking OK !')
			
