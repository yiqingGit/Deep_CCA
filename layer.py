import cPickle
import gzip
import os
import sys
import time

import copy
import numpy as np
import scipy.linalg as sl

from utils import *
from function import *

class Layer(object):
	def __init__(self,rng,n_in,n_out,activation=tanh,W=None,b=None, \
			base_lr = 0.01,dropoutFraction=0.0,local_decay=0.0005,momentum=0.9,regularization_type='L2'):
		if W is None:
			W_values = np.asarray(rng.uniform(
									low=-np.sqrt(6. / (n_in + n_out)),
									high=np.sqrt(6. / (n_in + n_out)),
									size=(n_in,n_out)),dtype=np.float32)
			if activation == sigmoid:
				W_values *=4
			W = W_values
		if b is None:
			b = np.zeros((n_out,),dtype=np.float32)

		self.W = W
		self.b = b
		self.grad_W =np.zeros((n_in,n_out),dtype=np.float32)
		self.grad_b = np.zeros((n_out,),dtype=np.float32)
		##
		self.activation = activation
		self.regularization_type = regularization_type
		self.dropoutFraction = dropoutFraction
		self.base_lr = base_lr
		self.local_decay = local_decay
		self.momentum = momentum
		self.activation = activation
		self.phrase =  'train'
		## params
		self.params = {}
		self.params['W'] = self.W
		self.params['b'] = self.b
		self.check_grad = False

	def _set_phrase(self,phrase='test'):
		self.phrase = phrase

	def _set_check_grad(self, check_grad=True):
		self.check_grad = check_grad

	def get_activation(self):
		return self.activation

	def _preSolve(self):
		self.history_grad_W = copy.deepcopy(self.grad_W)
		self.history_grad_b = copy.deepcopy(self.grad_b)

	def _forward(self,data = None):
		if(data == None):
			raise ValueError('data is None')
		lin_output = np.dot(data,self.W) + self.b
		self.output = (
						lin_output if self.activation is None 
						else self.activation(lin_output)
						) 
		## dropout
		if(self.dropoutFraction > 0):
			if(self.phrase == 'train'):
				if(self.check_grad is False):
					self.dropOutMask = np.double(np.random.rand(*self.output.shape) > self.dropoutFraction)
				self.output = self.output * self.dropOutMask
			else:
				#self.output = self.output * (1 - self.dropoutFraction)	  ## One of test format in dropout
				pass
		##
		return self.output

	def _backward(self, data=None, grad = None):
		'''
			data is the input value
		'''
		if(data == None or grad == None):
			raise ValueError('data or grad is None')
		## dropout
		if(self.dropoutFraction > 0):
			grad = grad * self.dropOutMask
		##
		grad_function = get_grad_activation(self.activation)
		if(grad_function != None):
			grad =  np.multiply(grad , grad_function(self.output))
		self.grad_W = np.dot(data.T, grad)
		self.grad_b = np.sum(grad, axis=0)
		grad_A = np.dot(grad, self.W.T)
		return grad_A

	def _computeUpdate(self,local_rate=None):
		if(local_rate == None):
			local_rate = self.base_lr
		if(self.regularization_type == 'L2'):
			self.grad_W = self.grad_W + self.local_decay * self.W
		self.grad_W = local_rate * self.grad_W + self.momentum * self.history_grad_W
		self.grad_b = local_rate * self.grad_b + self.momentum * self.history_grad_b  

	def _updateValue(self):
		self.W = self.W - self.grad_W
		self.b = self.b - self.grad_b
