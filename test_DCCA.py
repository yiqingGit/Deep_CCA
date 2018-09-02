import cPickle
import gzip
import os
import sys
import time

import copy
import numpy as np
import scipy.linalg as sl

from myIO import *
from utils import *
from DCCA import *
from evaluation import *

def DCCA_Net(n_in_1=None,n_in_2=None,n_hidden=None,n_out=None,layers=2,activation=relu,cost_function=cosin_cost, \
	base_lr = 0.01,dropoutFraction=0.0,local_decay=0.0005,momentum=0.9,regularization_type='L2'):
	if(n_out == None):
		raise ValueError('hidden layer node number must not be None!')
	##
	my_cmNet = DCCA(
		n_in_1  = n_in_1,
		n_in_2  = n_in_2,
		n_hidden = n_hidden,
		n_out  = n_out,
		layers = layers,
		activation= activation,
		cost_function = cost_function,
		base_lr = base_lr,
		dropoutFraction = dropoutFraction,
		local_decay = local_decay,
		momentum = momentum,
		regularization_type = regularization_type
	)
	return my_cmNet

def Train(my_cmNet,data1=None,data2=None,batch_size=None,iter_=None,max_iter=None,snapshot=None,snapshot_prefix=None,a=0.5, phase='pre_train'):
	if(data1 is None or data2 is None):
		raise ValueError('data1 or data2 must not be None!')
	my_cmNet.train_model(data1,data2,batch_size,iter_,max_iter,snapshot,snapshot_prefix,a, phase)	

def Test(my_cmNet,data1=None,data2=None,iter_=None,snapshot_prefix=None,trainFeat=False,obj_file1=None,obj_file2=None):
	if(data1 is None or data2 is None):
		raise ValueError('data1 or data2 must not be None!')
	##
	feat_1, feat_2 = my_cmNet.test_model(
		data1=data1,
		data2=data2,
		iter_=iter_,
		snapshot_prefix=snapshot_prefix,
		trainFeat=trainFeat
	)
	if(obj_file1 == None or obj_file2 == None):
		raise ValueError('obj_file can\'t be None !')
	write_matrix(feat_1,obj_file1)
	write_matrix(feat_2,obj_file2)
	return feat_1, feat_2
	
def check_NetGradientCorrect(layers=3,activation=relu,cost_function=euclid,base_lr = 0.01,dropoutFraction=0.5,local_decay=0.0005,momentum=0.9,regularization_type='L2'):
	"""
		Check Gradient in Net
	"""
	n_in_1 = 5
	n_in_2 = 6
	n_hidden = 10
	n_out = 6
	size = 20

	test_data1 = np.random.rand(size, n_in_1)
	test_data2 = np.random.rand(size, n_in_2)

	##
	my_test_cmNet = DCCA_Net(
		n_in_1  = n_in_1,
		n_in_2  = n_in_2,
		n_hidden = n_hidden,
		n_out  = n_out,
		layers = layers,
		activation= activation,
		cost_function = cost_function,
		base_lr = base_lr,
		dropoutFraction = dropoutFraction,
		local_decay = local_decay,
		momentum = momentum,
		regularization_type = regularization_type
	)
	my_test_cmNet.check_gradient(test_data1,test_data2)

def get_Train_Data():
	data_1_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/train_V.pkl'
	data_2_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/train_S.pkl'
	label_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/train_label.pkl'
	mean_1_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/mean_train_V.pkl'
	mean_2_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/mean_train_S.pkl'
	try:
		data1 = load_binary_data(data_1_file)
		data2 = load_binary_data(data_2_file)
		label = load_binary_data(label_file)
		mean1 = load_binary_data(mean_1_file)
		mean2 = load_binary_data(mean_2_file)
	except:
		data1 = load_matrix('/home/yiqing/DataSet/Corel30K/SplitData_300_300/Visual_Train.dat')
		data2 = load_matrix('/home/yiqing/DataSet/Corel30K/SplitData_300_300/Semantic_Train.dat')
		label = load_label('/home/yiqing/DataSet/Corel30K/SplitData_300_300/Label_Train.dat')
		mean1 = data1.mean(axis=0)
		mean2 = data2.mean(axis=0)
		data1 = data1 - np.tile(mean1, (data1.shape[0], 1))
		data2 = data2 - np.tile(mean2, (data2.shape[0], 1))
		save_binary_data(data1,data_1_file)
		save_binary_data(data2,data_2_file)
		save_binary_data(label,label_file)
		save_binary_data(mean1,mean_1_file)
		save_binary_data(mean2,mean_2_file)
	return data1, data2, label, mean1, mean2

def get_Test_Data(mean1,mean2):
	data_1_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/test_V.pkl'
	data_2_file='/home/yiqing/DataSet/Corel30K/SplitData_300_300/pkl/test_S.pkl'
	try:
		data1 = load_binary_data(data_1_file)
		data2 = load_binary_data(data_2_file)
	except: 
		data1 = load_matrix('/home/yiqing/DataSet/Corel30K/SplitData_300_300/Visual_Test.dat')
		data2 = load_matrix('/home/yiqing/DataSet/Corel30K/SplitData_300_300/Semantic_Test.dat')
		data1 = data1 - mean1
		data2 = data2 - mean2
		save_binary_data(data1,data_1_file)
		save_binary_data(data2,data_2_file)
	return data1, data2


def get_Train_NusWide():
	data_1_file='/home/yiqing/My_Workspace/split_300_300/pkl/train_V.pkl'
	data_2_file='/home/yiqing/My_Workspace/split_300_300/pkl/train_S.pkl'
	label_file='/home/yiqing/My_Workspace/split_300_300/pkl/train_label.pkl'
	mean_1_file='/home/yiqing/My_Workspace/split_300_300/pkl/mean_train_V.pkl'
	mean_2_file='/home/yiqing/My_Workspace/split_300_300/pkl/mean_train_S.pkl'
	try:
		data1 = load_binary_data(data_1_file)
		data2 = load_binary_data(data_2_file)
		label = load_binary_data(label_file)
		mean1 = load_binary_data(mean_1_file)
		mean2 = load_binary_data(mean_2_file)
	except:
		data1 = load_matrix('/home/yiqing/My_Workspace/split_300_300/Visual_Train.dat')
		data2 = load_matrix('/home/yiqing/My_Workspace/split_300_300/Semantic_Train.dat')
		label = load_label('/home/yiqing/My_Workspace/Data/class.dat')
		mean1 = data1.mean(axis=0)
		mean2 = data2.mean(axis=0)
		data1 = data1 - np.tile(mean1, (data1.shape[0], 1))
		data2 = data2 - np.tile(mean2, (data2.shape[0], 1))
		save_binary_data(data1,data_1_file)
		save_binary_data(data2,data_2_file)
		save_binary_data(label,label_file)
		save_binary_data(mean1,mean_1_file)
		save_binary_data(mean2,mean_2_file)
	return data1, data2, label, mean1, mean2

def get_Test_NusWide(mean1,mean2):
	data_1_file='/home/yiqing/My_Workspace/split_300_300/pkl/test_V.pkl'
	data_2_file='/home/yiqing/My_Workspace/split_300_300/pkl/test_S.pkl'
	try:
		data1 = load_binary_data(data_1_file)
		data2 = load_binary_data(data_2_file)
	except: 
		data1 = load_matrix('/home/yiqing/My_Workspace/split_300_300/Visual_Test.dat')
		data2 = load_matrix('/home/yiqing/My_Workspace/split_300_300/Semantic_Test.dat')
		data1 = data1 - mean1
		data2 = data2 - mean2
		save_binary_data(data1,data_1_file)
		save_binary_data(data2,data_2_file)
	return data1, data2

def mainTrain(DataSet='Corel30K'):
	current_path = os.getcwd()
	main_floder = os.path.dirname(current_path)
	if(DataSet == 'Corel30K'):
		data1, data2, label, mean1, mean2 =  get_Train_Data()
		main_floder = main_floder + '/Corel30K_Result_4096/'
	elif(DataSet == 'NusWide'):
		data1, data2, label, mean1, mean2 = get_Train_NusWide()
		main_floder = main_floder + '/NusWide_Result/'
	else:
		print('DataSet must be Corel30K or NusWide !')
		sys.exit()	

	if not os.path.exists(main_floder):
		os.makedirs(main_floder)

	size = data1.shape[0]
	n_hidden = 1000
	n_out = np.minimum(data1.shape[1], data2.shape[1])
	print data1.shape, data2.shape
	print (' net hidden layer node is : %i \n net feat layer node is : %i \n ' % (n_hidden,n_out))


	layers = 4
	base_lr = 0.01	
	activation = relu
	dropoutFraction = 0.0
	my_cmNet = DCCA_Net(
		n_in_1 = data1.shape[1],
		n_in_2 = data2.shape[1],
		n_hidden = n_hidden,
		n_out=n_out,
		layers = layers,
		activation=activation,
		cost_function=euclid,
		base_lr = base_lr,
		dropoutFraction = dropoutFraction
	)

	###
	batch_size = 1000
	snapshot = 1000

	isPreTrained = True
	#############################################
	a = 0.1
	ii = 1
	#
	snapshot_floder = main_floder +'/params/'
	if not os.path.exists(snapshot_floder):
		os.makedirs(snapshot_floder)
	snapshot_prefix = snapshot_floder + 'DCCA_layer4_'
	iter_ = 1
	pre_maxIter= 0
	max_iter = pre_maxIter + 4000

	###  Train ###
	Train(my_cmNet,data1=data1,data2=data2,batch_size=batch_size,iter_=iter_,max_iter=max_iter,snapshot=snapshot,snapshot_prefix=snapshot_prefix,a=a, phase='train')
	###  Test for Train Data ###
	iter_ = max_iter
	##
	feat_floder = main_floder +'/cca_feat/'
	if not os.path.exists(feat_floder):
		os.makedirs(feat_floder)
	train_obj_file1 = feat_floder + 'dcca_train_visual_' + str(iter_) + '.dat'
	train_obj_file2 = feat_floder + 'dcca_train_semantic_' + str(iter_) + '.dat'
	train_v, train_s = Test(my_cmNet,data1=data1,data2=data2,iter_=iter_,snapshot_prefix=snapshot_prefix,trainFeat=True,obj_file1=train_obj_file1,obj_file2=train_obj_file2)


	###  Test for Test Data ###	
	if(DataSet == 'Corel30K'):
		data1, data2 =  get_Test_Data(mean1, mean2)
	elif(DataSet == 'NusWide'):
		data1, data2 =  get_Test_NusWide(mean1, mean2)
	else:
		print('DataSet must be Corel30K or NusWide !')
		sys.exit()

	test_obj_file1 = feat_floder + 'dcca_test_visual_' + str(iter_) + '.dat'
	test_obj_file2 = feat_floder + 'dcca_test_semantic_' + str(iter_) + '.dat'
	
	test_v, test_s = Test(my_cmNet,data1=data1,data2=data2,iter_=iter_,snapshot_prefix=snapshot_prefix,trainFeat=False,obj_file1=test_obj_file1,obj_file2=test_obj_file2)
	##
	eval_floder = main_floder + '/evaluation/'
	if not os.path.exists(eval_floder):
		os.makedirs(eval_floder)

	##
	if(DataSet == 'Corel30K'):
		mainEval(test_v, train_s, iter_,ii,eval_floder,top=20,toptxtnum=4,dataset='Corel30K')
	elif(DataSet == 'NusWide'):
		mainEval(test_v, train_s, iter_,ii,eval_floder,top=20,toptxtnum=6,dataset='NusWide')
	else:
		print('DataSet must be Corel30K or NusWide !')
		sys.exit()

if __name__=='__main__':	
	#########################################################################
	# check_NetGradientCorrect()	
	#########################################################################
	mainTrain(DataSet='Corel30K')


	


		



