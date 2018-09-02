"""
	Utils for CCA and activation function
"""
import os
import numpy as np
import random
import cPickle
import scipy
import scipy.linalg as sl

class CCA_Layer(object):
	def __init__(self):
		## params
		self.mean_1 = None
		self.mean_2 = None
		self.W1 = None
		self.W2 = None
	
	def _cca(self,input_1, input_2, reg=1E-4):
		input_1 = np.array(input_1, dtype=np.double)
		input_2 = np.array(input_2, dtype=np.double)
		n1 , dim_1 = np.shape(input_1)
		n2 , dim_2 = np.shape(input_2)

		if not (n1==n2):
			print "multi-datasets samples not consistent..."
			return
		m = n1
		#print 'centered input dataset...'
		self.mean_1 = input_1.mean(axis=0)
		self.mean_2 = input_2.mean(axis=0)
		X1 = input_1 - self.mean_1
		X2 = input_2 - self.mean_2
	
		S11 = (np.dot(X1.T , X1) + reg * np.eye(dim_1)) / (m - 1.)
		S22 = (np.dot(X2.T , X2) + reg * np.eye(dim_2)) / (m - 1.)
		S12 = (np.dot(X1.T , X2)) / (m - 1.)

		sqrtm_S11 = sl.sqrtm(S11)
		sqrtm_S22 = sl.sqrtm(S22)
		# set _S12 = S11^{-1/2} S12 S22^{-1/2}
	    	# first multiply by S22^{-1/2} on right
		tmp = sl.solve(sqrtm_S22.T,S12.T).T
		# then multiply by S11^{-1/2} on left
		TT = sl.solve(sqrtm_S11 , tmp)
		# get SVD of S11^{-1/2} S12 S22^{-1/2}
		U, D, Vt = sl.svd(TT)
		
		self.W1 = np.array(sl.solve(sqrtm_S11, U),dtype=np.float32)
		self.W2 = np.array(sl.solve(sqrtm_S22, Vt.T),dtype=np.float32)

		# put S11^{-1/2} U  in _U
		_U  = sl.solve(sqrtm_S11 , U)
		# put Vt S22^{-1/2} in _Vt
		_Vt = sl.solve(sqrtm_S22.T , Vt.T).T
		# form _nabla12
		_nabla12 = np.dot(_U , _Vt)

		# View 1
		_nabla11 = -0.5 * np.dot(
			np.dot(_U , np.diag(D)),
			_U.T
		)
		D1 = np.dot(_nabla12 , X2.T)
		grad_X1 = (2. * np.dot(_nabla11,X1.T) + D1) 

	    	# View 2
		_nabla22 = -0.5 * np.dot(
			np.dot(_Vt.T, np.diag(D)),
			_Vt
		)
		D2 = np.dot(_nabla12.T , X1.T)
		grad_X2 = (2. * np.dot(_nabla22,X2.T) + D2) 

		corr = np.sum(D)
		k = np.minimum(dim_1,dim_2)
		cost = 0.5 * (m - 1.) * (k - corr) / m
		grad_X1 = - 0.5 * grad_X1 / m
		grad_X2 = - 0.5 * grad_X2 / m
		grad_X1 = np.array(grad_X1, dtype=np.float32)
		grad_X2 = np.array(grad_X2, dtype=np.float32)
		# print grad_X1, grad_X2
		return cost,D,grad_X1.T,grad_X2.T

	def _get_feat(self,input_1=None,input_2=None):
		if(input_1 is None or input_2 is None):
			raise ValueError("input can't be None !!!")
		## mean
		input_1 = (input_1 - self.mean_1)
		input_2 = (input_2 - self.mean_2)
		##
		feat_1 = np.dot(input_1 , self.W1)
		feat_2 = np.dot(input_2 , self.W2)
		return feat_1, feat_2

	def _load_model_param(self,f):
		self.mean_1 = cPickle.load(f)
		self.mean_2 = cPickle.load(f)
		self.W1 = cPickle.load(f)
		self.W2 = cPickle.load(f)

	def _save_model_param(self, f):
		cPickle.dump(self.mean_1, f)
		cPickle.dump(self.mean_2, f)
		cPickle.dump(self.W1, f)
		cPickle.dump(self.W2, f)

