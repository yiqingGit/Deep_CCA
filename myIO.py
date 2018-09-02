"""
	Utils for iostream
"""
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import scipy
import scipy.linalg as sl

def load_matrix(_file):
	f = open(_file)
	t = f.readline().strip('\r\n')
	feat = []	
	while t:
		sample = t.split()
		float_sample = []
		for x in sample:
			x = float(x)
			float_sample.append(x)
		feat.append(float_sample)
		t = f.readline().strip('\r\n')
	f.close()	
	feat = np.array(feat, dtype=np.float32)
	return feat

def load_label(_file):
	"""
		label is from 0 to (maximum_class -1)
	"""
	f = open(_file, 'r')
	label = []
	for line in f:
		data = line.strip('\r\n')
		data = float(data)
		label.append(int(data) - 1)
	f.close()
	label = np.array(label,dtype=np.float32)
	return label

def load_dic_word(_file):
	"""
		dic word is the dictionary word of dataset
	"""
	f = open(_file, 'r')
	dic_word= []
	for line in f:
		dic_word.append(line.strip('\r\n'))
	f.close()
	return dic_word

def load_dic_word_feat(_file):
	f = open(_file,'r')
	dic_word_feat = {}
	for line in f:
		i = line.strip().index(' ')
		word = line[:i]
		feat = line[i+1:]
		featArr = feat.split()
		floatArr = []
		for arr in featArr:
			floatArr.append(float(arr))
		floatArr = np.array(floatArr, dtype=np.float32)
		dic_word_feat[word] = floatArr
	f.close()
	return dic_word_feat

def write_matrix(data,_file):
	f = open(_file,'w')
	for row in data:
		vec = row
		for col in vec:
			f.write(str(col) + ' ')
		f.write('\n')
	f.close()	

def write_vector(data,_file):
	f = open(_file,'w')
	for vec in data:		
		f.write(str(vec) + '\n')
	f.close()

def save_binary_data(data,datafile):
	print('save data in pkl...')
	f = open(datafile,'wb')
	cPickle.dump(data,f)
	f.close()
	print('Done !')

def load_binary_data(datafile):
	try:
		print 'load src data from pkl...'
		f = open(datafile,'rb')
		data = cPickle.load(f)
		f.close()
		print 'success load !'
		return data
	except Exception, e:
		print('no pkl for load !')
		raise e	
