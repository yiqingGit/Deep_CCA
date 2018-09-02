"""
    Deep Net architecture
"""
import cPickle
import gzip
import os
import sys
import time

import copy
import numpy as np
import scipy.linalg as sl

from utils import *
from cca_utils import *
from function import *
from layer import *
from Noise import *

class ProgramArgs(object):
    def __init__(self, filename):
	f = open(filename, 'r')
	is_comments = False
	for line in f:
	    line = line.strip()
	    if(is_comments == False):
		if(line==None or line=='' or line.startswith('//')==True):
		    pass
		elif(line.startswith('/*') or line.startswith('/**')):
		    is_comments = True
		elif(line.find('n_in') != -1):
		    id_ = line.rindex('=')
		    self.n_in = int(line[(id_+1):])
		elif(line.find('n_hidden') != -1):
		    id_ = line.rindex('=')
		    self.n_hidden = int(line[(id_+1):])
		elif(line.find('n_out') != -1):
		    id_ = line.rindex('=')
		    self.n_out = int(line[(id_+1):])
		else:
		    raise ValueError('Have not implements yet')
	    else:
		if(line.startswith('*/') or line.startswith('**/')):
		    is_comments = False
	f.close()
	
class NeuralNet(object):
    def __init__(self,n_in,n_hidden,n_out,layers=3,activation=tanh, \
            base_lr = 0.01,dropoutFraction = 0.0,local_decay=0.0005,momentum=0.9,regularization_type='L2'):
        if(layers <2):
            raise ValueError('error layes input!')
        ##      
        self.data = None
        self.label = None
        self.layers = layers
        ###############################################################
        self.net_layers = {}
        self.net_params = {}
        ## Layer
        rng = np.random.RandomState(1234)
	print('======================== Layer Format ========================')
        for layer in xrange(1,self.layers + 1,1):
            # Input Layer
            if(layer == 1):
                layer_in = n_in
            else:
                layer_in = n_hidden
            # Output Layer
            if(layer == self.layers):
                layer_out = n_out
                layer_activation = activation
		layer_dropoutFraction = 0.0
            else:
                layer_out = n_hidden
                layer_activation = activation
		layer_dropoutFraction = dropoutFraction
            #############  
            hiddenLayer = Layer(
                rng = rng,
                n_in = layer_in,
                n_out = layer_out,
                activation = layer_activation,
                base_lr = base_lr,
		dropoutFraction = layer_dropoutFraction,
                local_decay = local_decay,
                momentum = momentum,
                regularization_type = regularization_type
            )
            self.net_layers[layer] = hiddenLayer
            self.net_params[layer] = self.net_layers[layer].params
	    self.phrase = 'train'
            print("layer_in :{} -- layer_out:{}".format(layer_in, layer_out))
	print('====================== Parameter Setting =====================')
	print('[base_lr ]: {}'.format(base_lr))
	print('[local_decay ]: {}'.format(local_decay))
	print('[momentum ]: {}'.format(momentum))
	print('[activation ]: {}'.format(activation))
	print('[dropoutFraction ]: {}'.format(dropoutFraction))
        ###############################################################

    def get_output(self):
        return self.net_layers[self.layers].output 
	
    def set_phrase(self,phrase='test'):
	if(phrase == self.phrase):
		return
	##
	self.phrase = phrase
	for layer in xrange(1, self.layers+1, 1):
		self.net_layers[layer]._set_phrase(phrase)

    def set_check_grad(self,check_grad=True):
	for layer in xrange(1, self.layers+1, 1):
		self.net_layers[layer]._set_check_grad(check_grad)

    def set_input(self,data=None, label=None):
        if(data == None):
            raise ValueError('no input data!')
        self.data = data  
        self.label = label

    def _preSolve(self):
        """
            preSolve
        """
        ## Layer
        for layer in xrange(1,self.layers + 1,1):
            self.net_layers[layer]._preSolve()

    def _forward(self):
        """
            forward 
        """
        if self.data is None: 
            raise ValueError('input must not be None!')
        ## Layer
        output = self.data
        for layer in xrange(1,self.layers + 1,1):
            output = self.net_layers[layer]._forward(output)

    def _backward(self,grad=None):
        """
            backward
        """
        if grad is None: 
            raise ValueError('grad must not be None!')
        ## Layer 
        for layer in xrange(self.layers,0,-1):
	    if(layer == 1):
		data = self.data
	    else:
		data = self.net_layers[layer-1].output
            grad = self.net_layers[layer]._backward(data, grad)

    def _updateValue(self,local_rate=None):
        """
            updateValue
        """
        if(local_rate == None):
            raise ValueError('local_rate can\'t be None !')
        ## Layer
        for layer in xrange(1,self.layers + 1,1):
            self.net_layers[layer]._computeUpdate(local_rate)
            self.net_layers[layer]._updateValue()

    def _load_model_param(self,f):
        ## Hidden Layer
        for layer in xrange(1,self.layers + 1,1):
            self.net_layers[layer].W = cPickle.load(f)
            self.net_layers[layer].b = cPickle.load(f)

    def _save_model_param(self, f):
        ## Layer
        for layer in xrange(1,self.layers + 1,1):
            cPickle.dump(self.net_layers[layer].W, f)
            cPickle.dump(self.net_layers[layer].b, f)

class DCCA(object):
    def __init__(self,n_in_1,n_in_2,n_hidden,n_out,layers=4,activation=tanh,cost_function=euclid, \
        base_lr = 0.01,dropoutFraction=0.0,gamma=0.001,power=0.75,local_decay=0.0005,momentum=0.9,regularization_type='L2'):
        if(layers < 3):
            raise ValueError("layers is invalid !")
        ## Net
	print('######################## Net Setting #########################')
        self.net1 = NeuralNet(
            n_in = n_in_1,
            n_hidden = n_hidden,
            n_out = n_out,
            layers=layers,
            activation = activation,
            base_lr = base_lr,
	    dropoutFraction = dropoutFraction,
            local_decay = local_decay,
            momentum = momentum,
            regularization_type = regularization_type
        )
	print('######################## Net Setting #########################')
        self.net2 = NeuralNet(
            n_in = n_in_2,
            n_hidden = n_hidden,
            n_out = n_out,
            layers=layers,
            activation = activation,
            base_lr = base_lr,
	    dropoutFraction = dropoutFraction,
            local_decay = local_decay,
            momentum = momentum,
            regularization_type = regularization_type
        )
	self.cca_layer = CCA_Layer()
	##
        self.base_lr = base_lr
        self.gamma = gamma
        self.power = power
        self.cost_function = cost_function
	print('##############################################################\n')

    def set_input(self,data1=None, data2=None, label=None):
	self.net1.set_input(data1, label)
	self.net2.set_input(data2, label)

    def get_output(self):
        net1_output = self.net1.get_output()
        net2_output = self.net2.get_output()
        return net1_output,net2_output

    def set_phrase(self,phrase='test'):
        self.net1.set_phrase(phrase)
        self.net2.set_phrase(phrase)

    def set_check_grad(self,check_grad=True):
	self.net1.set_check_grad(check_grad)
	self.net2.set_check_grad(check_grad)

    def preSolve(self):
        self.net1._preSolve()
        self.net2._preSolve()

    def forward(self):      
        self.net1._forward()
        self.net2._forward()

    def backward(self,grad1, grad2):
        self.net1._backward(grad1)
        self.net2._backward(grad2)
        
    def get_cost(self,a = 1):
        n = self.net1.data.shape[0]
	##
        H1, H2 = self.get_output()
        loss, corr, grad_H1, grad_H2 = self.cca_layer._cca(H1, H2) 
	total_corr = np.sum(corr)
        ##
	print("[ %s ] Total Canonical Correlation = %.5f" % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), total_corr))
        return loss, grad_H1, grad_H2

    def update(self,local_rate):
        self.net1._updateValue(local_rate)
        self.net2._updateValue(local_rate)

    def getLearningRate(self,iter_):
        rate = self.base_lr * pow(1 + self.gamma * iter_ , - self.power)
        print("[ %s ] Iteration %d , lr = %.8f " % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),iter_, rate))        
        return rate

    def pre_cost(self,a=0.0):
	n = self.net1.data.shape[0]
        ##
        loss1, grad1 = softmax(self.net1.get_output(), self.net1.label)
        loss2, grad2 = softmax(self.net2.get_output(), self.net2.label)
	print("[ %s ]  net1_loss = %.5f, net2_loss=%.5f " % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), loss1, loss2))
        loss = loss1 + loss2
	H1, H2 = self.get_feat()
	grad_H1 = np.zeros_like(H1)
	grad_H2 = np.zeros_like(H2)
        return loss, grad1, grad2, grad_H1, grad_H2

    def pre_train(self,local_rate,a=0.5):
	self.preSolve()
	self.forward()
	loss, grad1, grad2, grad_H1, grad_H2 = self.pre_cost(a)
        self.backward(grad1, grad2, grad_H1, grad_H2)
        self.update(local_rate)
        return loss

    def train(self,local_rate,a=0.5):
        if(self.net1.data == None or self.net2.data ==None):
            raise ValueError('no input data!')
        self.preSolve()
        self.forward()
        loss, grad1, grad2 = self.get_cost(a)
        self.backward(grad1, grad2)
        self.update(local_rate)
        return loss

    def train_model(self,data1=None,data2=None,batch_size=100,iter_ = 1,max_iter = 1000,snapshot=100,snapshot_prefix='param',a=0.5, phase='train'):
        if(data1 is None or data2 is None):
            raise ValueError('no input data!')
        ##
	self.set_phrase(phrase='train')
        size = data1.shape[0]
        n_batches = size / batch_size
        if(size % batch_size !=0):
            n_batches +=1

        batch_index = 0
        loss = 0.

        try:
            self.load_model_param(snapshot_prefix+'_Iter_' + str(iter_) + '.snapshot')
	    print("load model param from %s_Iter_%d.snapshot done ..." % (snapshot_prefix,iter_))
        except Exception, e:
            print('there is no trained model in iter {}'.format(iter_))

        while(iter_ <= max_iter):
            ## snapshot
            if(iter_ % snapshot == 0):
                self.save_model_param(snapshot_prefix+'_Iter_' + str(iter_) + '.snapshot')
            ##
            if batch_index == 0:
                random_batch = randperm(size,size)
            if (size % batch_size !=0) and ((size-batch_index*batch_size) < batch_size):
                #print('{} -> {}' .format((size-batch_size) , size))
                batch_data_1 = compRandomBatchData(data1, random_batch[(size-batch_size) : size])
                batch_data_2 = compRandomBatchData(data2, random_batch[(size-batch_size) : size])
            else:
                #print('{} -> {}' .format((batch_index*batch_size) , (batch_index+1)*batch_size))
                batch_data_1 = compRandomBatchData(data1, random_batch[batch_index*batch_size : (batch_index+1)*batch_size])
                batch_data_2 = compRandomBatchData(data2, random_batch[batch_index*batch_size : (batch_index+1)*batch_size])
            ##
            self.set_input(data1=batch_data_1, data2=batch_data_2)
	    #
	    ## The Whole DataSet Training ##
	    # self.set_input(data1=data1, data2=data2)
	    ##
            local_rate = self.getLearningRate(iter_)
            ##
	    loss = self.train(local_rate,a)
            ## Display
            print("[ %s ]     Train net ouput iter:%d  loss = %.8f  (*1  =   %.8f [phase = %s])" % 
                    (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),iter_,loss,loss, phase))
            ##
            batch_index +=1
            # Reset the batch_index
            if ((size - batch_index * batch_size) <= 0):
                batch_index = 0
            iter_ +=1
        ##
        self.forward()
        loss, grad1, grad2 = self.get_cost()
        print ("[ %s ] Optimization Done. Train net loss: %.8f (*1 = %.8f)" % ( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),loss,loss))  

    def test_model(self,data1=None,data2=None,iter_ = 1,snapshot_prefix='param', trainFeat=False):
        if(data1 is None or data2 is None):
		raise ValueError('data1 or data2 must not be None!')
	##
	self.set_phrase(phrase='test')
	self.load_model_param(snapshot_prefix+'_Iter_' + str(iter_) + '.snapshot')
	self.set_input(data1=data1,data2=data2)
	self.forward()
	H1, H2 = self.get_output()
	if(trainFeat is True):
	   loss, corr, grad_H1, grad_H2 = self.cca_layer._cca(H1, H2)
	   main_floder = os.path.dirname(os.path.dirname(snapshot_prefix))
	   write_vector(corr, main_floder + '/' + 'Corel30K_DCCA_corr_'+str(iter_)+'.dat')
	   self.save_model_param(snapshot_prefix+'_Iter_' + str(iter_) + '.snapshot')
	feat_1, feat_2 = self.cca_layer._get_feat(H1, H2)
	return feat_1, feat_2

    def check_gradient(self,data1,data2,label=None,epsilon = 1e-4,er = 1e-5):
        if(label !=None and label.ndim == 1):
            label  = label2dim(label , self.n_label)
        self.set_input(data1,data2,label)
        self.forward()
        loss, grad1, grad2 = self.get_cost()
        self.backward(grad1, grad2)
	self.set_check_grad(True)
        for layer in xrange(1,self.net1.layers + 1,1):
            W = self.net1.net_layers[layer].W
            grad_W = self.net1.net_layers[layer].grad_W
            grad_b = self.net1.net_layers[layer].grad_b
            rows, cols = W.shape
            for i in xrange(rows):
                for j in xrange(cols):
                    W_p = copy.deepcopy(W)
                    W_p[i][j] += epsilon                    
                    self.net1.net_layers[layer].W = W_p
                    self.forward()
                    loss_p, grad1_p, grad2_p = self.get_cost()

                    W_m = copy.deepcopy(W)
                    W_m[i][j] -= epsilon
                    self.net1.net_layers[layer].W = W_m
                    self.forward()
                    loss_m, grad1_m, grad2_m= self.get_cost()

                    grad_pm = (loss_p - loss_m) / (2. * epsilon)
                    e = np.abs(grad_pm - grad_W[i][j])
                    print('e : {} \t grad_pm : {}  -- grad_W: {}  -- grad_b:{} '.format(e,grad_pm,grad_W[i][j],grad_b[j]))
                    #assert(e < er)
                    if(e > er):
                        print('layer %d (row:%d , col:%d ) numerical gradient checking failed !' % (layer,i,j))
                        break
                    else:
                        print('layer %d (row:%d , col:%d ) numerical gradient checking OK !' % (layer,i,j))
        	    ## Recover W
		    self.net1.net_layers[layer].W = W
      
    def load_model_param(self,param_file):
        f = open(param_file,'rb')
        ## Net1 Net2, CCA_Layer
        self.net1._load_model_param(f)
        self.net2._load_model_param(f)
	self.cca_layer._load_model_param(f)
        f.close()

    def save_model_param(self, param_file):
        f = open(param_file,'wb')
        ## Net1, Net2, CCA_Layer
        self.net1._save_model_param(f)
        self.net2._save_model_param(f)	
	self.cca_layer._save_model_param(f)	
        f.close()
