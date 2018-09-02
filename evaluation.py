#
import numpy as np
import cPickle

from data_process import *

def load_feat(data_file):
	"""
		load the feature
	"""
	f = open(data_file,'rb')
	feat = cPickle.load(f)
	f.close()
	return feat

def load_data(_file):
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

def load_label(label_file):
	"""
		load the label
	"""
	f = open(label_file , 'r')
	labellist = []
	for label in f:
		labellist.append(int(float(label.strip('\n'))))
	f.close()
	labellist = np.array(labellist)
	return labellist

def load_text(textfile):
	"""
		load the text
	"""
	f = open(textfile , 'r')
	textlist = []
	for text_ in f:
		words = text_.strip('\n').split()
		textlist.append(words)
	f.close()
	textlist = np.array(textlist)
	return textlist

def write_text(textfile,textlist):
	"""
		write the text
	"""
	f = open(textfile , 'w')
	for words in textlist:
		for x in xrange(len(words)):
			if(x !=len(words) - 1):
				f.write(words[x] + ' ')
			else:
				f.write(words[x])
		f.write('\n')
	f.close()

def write_prec(precfile,preclist):
	"""
		write the text
	"""
	f = open(precfile , 'w')
	for prec in preclist:		
		f.write(str(prec) + '\n')
	f.close()

def norm_mat(mat):
	"""
		vec is a vector 1 x dim
	"""
	square_mat = mat * mat	
	norm_factor = np.sum(square_mat , axis=1)
	len_mat = np.sqrt(norm_factor)	
	dlen_mat = np.divide(1. , len_mat)
	norm_mat = mat * np.tile(dlen_mat , (mat.shape[1] , 1)).T
	return norm_mat

def cosinScore(testVec, trainFeat):
	"""
		cosinScore is compute the score between test 
		feature vector and all train feature,
		return the score list
	"""
	n = trainFeat.shape[0]
	norm_testVec = testVec / np.sqrt((testVec * testVec).sum())
	norm_trainFeat = norm_mat(trainFeat)
	norm_repeat_testVec = np.tile(norm_testVec,(n,1))
	scorelist = (norm_repeat_testVec * norm_trainFeat).sum(axis=1)	
	return scorelist

def euclidScore(testVec, trainFeat):
	"""
		euclidScore is compute the score between test 
		feature vector and all train feature,
		return the score list
	"""
	n = trainFeat.shape[0]
	repeat_testVec = np.tile(testVec,(n,1))
	t = repeat_testVec - trainFeat
	scorelist = np.sum(t * t , axis=1)
	return scorelist

def get_precision_recall(reswords, words):
	"""
		compute the precision and recall
	"""
	#print('The true tag: \t {}'.format(words))
	hit = 0.
	hit += len(reswords) + len(words) - len(set(reswords + words)) 
	prec = hit / (len(reswords)+1e-8)
	recall = hit / (len(words)+1e-8)
	return prec, recall

def getTopScoreImage(scorelist,train_Idxlist, labellist, top,dist_function):
	"""
		getTopScoreImage: use the scorelist 
		and labellist to get the top score label and text
	"""
	scoreTuple = []
	for i in xrange(len(scorelist)):
		tuple_ = (train_Idxlist[i], scorelist[i])
		scoreTuple.append(tuple_) 
	if(dist_function==cosinScore):
		scoreTuple = sorted(scoreTuple , cmp = lambda x, y: cmp(x[1],y[1]), reverse=True)
	elif(dist_function==euclidScore):
		scoreTuple = sorted(scoreTuple , cmp = lambda x, y: cmp(x[1],y[1]), reverse=False)
	#print('scorelist :{}' ,format(scoreTuple[0:100]))
	#print len(scoreTuple)
	reslist = []
	resIdx = []
	for i in xrange(top):
		topIdx = scoreTuple[i][0]  
		resIdx.append(topIdx)
		reslist.append(labellist[topIdx])
		#print('idx:{} -- label:{}'.format(topIdx, labellist[topIdx]))
	return resIdx, reslist

def getTopScoreText(resIdx, textfile, toptxtnum=5):
	textlist = load_text(textfile)
	textdict = {}
	for idx in resIdx:
		text = textlist[idx]
		for t in text:
			if(textdict.has_key(t)):
				textdict[t] = textdict[t] + 1
			else:
				textdict[t] = 1
	textdict = sorted(textdict.iteritems() , key = lambda x:x[1], reverse=True)
	reswords = []
	if(toptxtnum > len(textdict)):
		toptxtnum = len(textdict)
	for i in xrange(toptxtnum):
		key_value = textdict[i]
		word = key_value[0]
		count_ = key_value[1]
		reswords.append(word)
	#print('recommend tag: \t {}'.format(reswords))
	return reswords

def separateTrainTest(labelfile, textfile, train_Idx_file, test_Idx_file):	
	test_Idxlist = load_label(test_Idx_file)
	train_Idxlist = load_label(train_Idx_file)
	labellist = load_label(labelfile)
	textlist = load_text(textfile)
	return train_Idxlist,test_Idxlist,labellist,textlist

def getTest_textlist(textlist,test_Idxlist):
	test_textlist = []
	for i in xrange(len(test_Idxlist)):
		test_idx = test_Idxlist[i]
		words = textlist[test_idx]
		test_textlist.append(words)
	return test_textlist

def generate_dic(wordslist):
	"""
		key: word ,value:[imgNo] ,Example: {'apple':[1,2,3,4,5]}
	"""
	dic = {}
	for i in xrange(len(wordslist)):
		words = wordslist[i]
		for j in xrange(len(words)):
			w = words[j]
			arr = []
			if(dic.has_key(w)):
				arr = dic[w]
			else:
				arr = []
			arr.append(i)
			dic[w] = arr
			#print dic
	return dic

def get_dic(reswordslist,test_textlist):	
	re_dic = generate_dic(reswordslist)
	gt_dic = generate_dic(test_textlist)
	return re_dic,gt_dic

def get_per_class_recall(re_dic,gt_dic):
	"""
		compute the per class precision
	"""
	gt_keys = gt_dic.keys()
	pc_recall = 0.
	for key in gt_keys:
		gt_val = gt_dic[key]
		ground_Img = float(len(gt_val))
		correct_Img = 0.
		if(re_dic.has_key(key)):
			re_val = re_dic[key]			
			correct_Img = float(len(re_val) + ground_Img - len(set(re_val + gt_val)))
		pc_recall += correct_Img / ground_Img
	avg_pc_recall = pc_recall / len(gt_keys)
	#print len(gt_keys)
	return avg_pc_recall

def get_per_class_precision(re_dic,gt_dic):
	re_keys = re_dic.keys()
	pc_precision = 0.
	for key in re_keys:
		re_val = re_dic[key]
		correct_Img = 0.
		pred_Img = float(len(re_val))
		if(gt_dic.has_key(key)):
			gt_val = gt_dic[key]
			correct_Img =  float(pred_Img + len(gt_val) - len(set(re_val + gt_val)))
		pc_precision += correct_Img / pred_Img
	avg_pc_prec = pc_precision / len(re_keys)
	#print len(re_keys)
	return avg_pc_prec

def get_overall_recall(re_dic,gt_dic):
	gt_keys = gt_dic.keys()
	overall_recall = 0.
	all_correct_Img = 0.
	all_ground_Img = 0.
	for key in gt_keys:
		gt_val = gt_dic[key]
		ground_Img = float(len(gt_val))
		correct_Img = 0.
		if(re_dic.has_key(key)):
			re_val = re_dic[key]
			correct_Img = float(len(re_val) + ground_Img - len(set(re_val + gt_val)))
		all_correct_Img += correct_Img
		all_ground_Img += ground_Img
	overall_recall = all_correct_Img / all_ground_Img
	return overall_recall

def get_overall_precision(re_dic,gt_dic):
	re_keys = re_dic.keys()
	overall_prec = 0.
	all_correct_Img = 0.
	all_pred_Img = 0.
	for key in re_keys:
		re_val = re_dic[key]
		pred_Img = float(len(re_val))
		correct_Img = 0.
		if(gt_dic.has_key(key)):
			gt_val = gt_dic[key]
			correct_Img = float(pred_Img + len(gt_val) - len(set(re_val + gt_val)))
		all_correct_Img += correct_Img
		all_pred_Img += pred_Img
	overall_prec = all_correct_Img / all_pred_Img
	return overall_prec

def evaluation_protocol(train_Idx_file,test_Idx_file,labelfile,textfile,reswordslist=None,reswordslist_file=None):
	################ Another Revaluation ##################	
	if(reswordslist == None):
		reswordslist = load_text(reswordslist_file)
	train_Idxlist,test_Idxlist,labellist,textlist = \
		separateTrainTest(labelfile, textfile, train_Idx_file, test_Idx_file)
	test_textlist = getTest_textlist(textlist,test_Idxlist)
	########
	re_dic,gt_dic = get_dic(reswordslist,test_textlist)
	avg_pc_recall = get_per_class_recall(re_dic,gt_dic)
	avg_pc_prec = get_per_class_precision(re_dic,gt_dic)
	overall_recall = get_overall_recall(re_dic,gt_dic)
	overall_prec = get_overall_precision(re_dic,gt_dic)
	print('avg_pc_recall is : %.8f' % (avg_pc_recall))
	print('avg_pc_prec is : %.8f' % (avg_pc_prec))
	print('overall_recall is : %.8f' % (overall_recall))
	print('overall_prec is : %.8f' % (overall_prec))
	return avg_pc_recall,avg_pc_prec,overall_recall,overall_prec
	########################

def v2s_retrieval(test_v ,test_Idx_file, train_s, train_Idx_file, \
	labelfile, textfile, top=100, toptxtnum=5,dist_function=cosinScore):
	"""
		- test_v is the visual feature of test samples
		- test_label_file is the label file of test samples
		- train_s is the  semantic feature of train samples
		- train_label_file is the label file of train samples
		- textfile is the text file of the all sample include test and train
	"""
	train_Idxlist,test_Idxlist,labellist,textlist = \
		separateTrainTest(labelfile, textfile, train_Idx_file, test_Idx_file)
	preclist = []
	recalllist = []
	reswordslist = []
	test_textlist = getTest_textlist(textlist,test_Idxlist)
	for i in xrange(len(test_v)):
		print i
		vecfeat = test_v[i,:]		
		if(dist_function==cosinScore):
			scorelist = cosinScore(vecfeat, train_s)
		elif(dist_function==euclidScore):
			scorelist = euclidScore(vecfeat, train_s)
		resIdx, reslist = getTopScoreImage(scorelist=scorelist,train_Idxlist=train_Idxlist, labellist=labellist,top=top,dist_function=dist_function)
		reswords = getTopScoreText(resIdx, textfile, toptxtnum)
		reswordslist.append(reswords)
		############# Revaluation  #############
		test_idx = test_Idxlist[i]
		words = textlist[test_idx]
		prec, recall = get_precision_recall(reswords, words)
		print('img %d retrieve semantic precision is : %.8f, recall is : %.8f' % (i, prec, recall))
		preclist.append(prec)
		recalllist.append(recall)
	################ Another Revaluation ##############
	#avg_pc_recall,avg_pc_prec,overall_recall,overall_prec = \
	#	evaluation_protocol(train_Idx_file,test_Idx_file,labelfile,textfile,reswordslist)
	###################################################	
	avgprec = sum(preclist) / len(preclist)
	avgrecall = sum(recalllist) / len(recalllist)
	avgF = (2. * avgprec * avgrecall) / (avgprec + avgrecall)
	print('all test img [num: %d] retrieve semantic precision is : %.8f, recall is : %.8f, F-score is : %.8f' \
		% (len(preclist), avgprec, avgrecall, avgF))
	return preclist,recalllist, reswordslist

def s2v_retrieval(test_s ,test_Idx_file, \
		train_v, train_Idx_file, \
		labelfile, textfile, top=100):
	"""
	"""
	##
	train_Idxlist,test_Idxlist,labellist,textlist = \
		separateTrainTest(labelfile, textfile, train_Idx_file, test_Idx_file)
	preclist = []
	for i in xrange(len(test_s)):
		vecfeat = test_s[i,:]
		idx = test_Idxlist[i]
		label = labellist[idx]
		scorelist = cosinScore(vecfeat, train_v)
		resIdx, reslist = getTopScoreLabel(scorelist,train_Idxlist, labellist, top)
		prec = precision(reslist, label)		
		preclist.append(prec)
		print('img %d retrieve semantic precision is : %.8f' % (i, prec))
	avgprec = sum(preclist) / len(preclist)
	print('all test img [num: %d] retrieve semantic precision is : %.8f' \
		% (len(preclist), avgprec))
	return preclist, avgprec

def mainEval(test_v,train_s,iter_,ii,eval_floder,top = 100,toptxtnum = 5, dataset='Corel30K'):
	dist_function=cosinScore
	train_prefix = 'train'
	test_prefix = 'test'
	##
	if(dataset == 'Corel30K'):
		print('Corel30K...')
		labelfile = '/home/yiqing/Corel_30K/class.dat'
		textfile = '/home/yiqing/Corel_30K/WordsIndex.txt'
		train_Idx_file = '/home/yiqing/Corel_30K/Set_10_0_Train.index'
		test_Idx_file = '/home/yiqing/Corel_30K/Set_10_0_Test.index'
	elif(dataset == 'NusWide'):
		print('NusWide...')
		labelfile = '/home/yiqing/My_Workspace/Data/class.dat'
		textfile = '/home/yiqing/My_Workspace/index/Tags_Result.txt'
		train_Idx_file = '/home/yiqing/My_Workspace/index/TrainIndex.txt'
		test_Idx_file = '/home/yiqing/My_Workspace/index/TestIndex.txt'		
	else:
		raise ValueError('Please point out dataset(Corel30K or NusWide) !!!')
	##
	print('=============   Visual search Semantic   =============')
	print('test_v shape %d x %d' %(test_v.shape))
	print('train_s shape %d x %d' %(train_s.shape))
	preclist,recalllist, reswordslist = v2s_retrieval(test_v ,test_Idx_file,train_s, \
		train_Idx_file,labelfile, textfile,top,toptxtnum,dist_function=dist_function)

	train_Idxlist,test_Idxlist,labellist,textlist = \
		separateTrainTest(labelfile, textfile, train_Idx_file, test_Idx_file)
	test_Textlist = getTest_textlist(textlist,test_Idxlist)
	##
	#eval_floder = '/home/yiqing/experiment/result/evaluation'
	write_text(eval_floder + '/corr'+ str(ii)+'_iter_'+str(iter_)+'_semantic.text',test_Textlist)

	write_prec(eval_floder + '/corr'+ str(ii)+'_iter_'+str(iter_)+'_prec.dat',preclist)
	write_prec(eval_floder + '/corr'+ str(ii)+'_iter_'+str(iter_)+'_recall.dat',recalllist)
	write_text(eval_floder + '/corr'+ str(ii)+'_iter_'+str(iter_)+'_text.dat',reswordslist)
	print('=============   V2S search Done   =============')

