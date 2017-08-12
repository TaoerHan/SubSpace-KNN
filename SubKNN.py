'''
	SubDistance KNN
	author AIbert
	hantaoer@foxmail.com
'''

import numpy as np
import numba

class SubKnnClassifier():
	"""docstring for SubKnn"""

	def __init__(self, k, threshold=0):
		if(k>0 and k==int(k)): self.k = k
		else:	print('k should be positive integer')
		self.threshold = threshold
		self.X = None; self.y=None
		self.class_weight = {}

	def modify_param(self, k=0, class_weight=0, threshold=0):
		if(k):	self.k = k
		if(class_weight):	self.class_weight = class_weight
		if(threshold):	self.threshold = threshold

	def fit(self, train, label=None):
		self.X = train
		if(label is not None):
			self.y = label
			self.class_weight = [np.sum(self.y==i)/len(self.y) 
									for i in set(self.y)]
			self.class_weight = dict(zip(set(self.y), self.class_weight))

	@numba.jit
	def predict(self, test):
		if(test.shape[1] !=self.X.shape[1]):
			print('train and test features dimension not matched')
			return None
		pl = np.zeros(len(test))
		cnt = 0
		fromkeys = dict.fromkeys;sum = np.sum
		for i in iter(test):
			trainX = self.X.T[i>self.threshold].T
			# SubDistance
			t = i[i>self.threshold]
			distance = sum((trainX - t)**2, axis=1)
			# getTop k of an array
			ind = np.argsort(distance)[0:self.k]
			weight = 1/(distance[ind]+1);	yi = self.y[ind]
			weight /= np.sum(weight)
			lable_d = fromkeys(yi,0)
			for k,v in zip(yi,weight):
				lable_d[k] += v

			s = -1; p = -1
			for k,v in lable_d.items():
				v /= self.class_weight[k]
				if(s<v):
					s = v; p = k
			pl[cnt] = p;cnt += 1
		return pl

	@numba.jit
	def predict_proba(self, test):
		if(test.shape[1] !=self.X.shape[1]):
			print('train and test features dimension not matched')
			return None
		pl = np.zeros((len(test),len(self.class_weight)))
		cnt = 0
		fromkeys = dict.fromkeys;sum = np.sum
		for i in iter(test):
			trainX = self.X.T[i>self.threshold].T
			# SubDistance
			t = i[i>self.threshold]
			distance = sum((trainX - t)**2, axis=1)
			ind = np.argsort(distance)[0:self.k]
			weight = 1/(distance[ind]+1); yi = self.y[ind]
			weight /= np.sum(weight)
			lable_d = fromkeys(yi,0)
			for k,v in zip(yi,weight):
				lable_d[k] += v

			loc = 0
			for k,v in lable_d.items():
				v /= self.class_weight[k]
				pl[cnt][loc] = v;loc+=1
			cnt += 1
		return pl

	@numba.jit
	def Kneighbors(self, test, return_distance=False):
		Kneighbors_ind = np.zeros((len(test),self.k)).astype(int)
		cnt = 0
		sum = np.sum;argsort = np.argsort
		if(return_distance is False):
			for i in iter(test):
				trainX = self.X.T[i>self.threshold].T
				t = i[i>self.threshold]
				distance = sum((trainX - t)**2, axis=1)
				Kneighbors_ind[cnt,:] = argsort(distance)[0:self.k]
				cnt += 1
			return Kneighbors_ind
		else:
			Kneighbors_distance = np.zeros((len(test),self.k))
			for i in iter(test):
				trainX = self.X.T[i>self.threshold].T
				t = i[i>self.threshold]
				distance = sum((trainX - t)**2, axis=1)
				Kneighbors_ind[cnt,:] = argsort(distance)[0:self.k]
				Kneighbors_distance[cnt,:] = distance[Kneighbors_ind[cnt,:]]
				cnt += 1
			return Kneighbors_ind,Kneighbors_distance


if __name__ == '__main__':
	
	from SubKNN import SubKnnClassifier
	import numpy as np
	import matplotlib.pyplot as plt
	import time

	np.random.seed(0)
	s = 200
	data = np.random.rand(s,3)*10
	data[int(s/2):] = data[int(s/2):]+8
	data[-int(s/4):,2] = 0

	y = np.array([0]*int(s/2) + [1]*int(s/4) + [2]*int(s/4))
	
	ind = np.random.randint(0,3,size=s)
	tr = data[ind!=0]
	tr_y = y[ind!=0]
	te = data[ind==0]
	te_y = y[ind==0]

	print("train samples ",len(tr_y),"-- test samples",len(te_y),"-- K =",5)
	T = time.clock()
	clf = SubKnnClassifier(k=5,threshold=-1)
	clf.fit(tr,tr_y)
	# clf.modify_param(k=5,class_weight={0:0.5,1:0.3,2:0.1},threshold=1)
	py = clf.predict(te)
	probay = clf.predict_proba(te)
	print(probay[0:5])
	print(clf.class_weight)
	print('Time',time.clock()-T,"ACC: ", sum(py==te_y)/len(py))


# ==========================
# 1 --------------
# @jit
# train samples  26677 -- test samples 13323 -- K = 10
# ACC:  1.0
# [Finished in 114.5s]
# 2 --------------
# train samples  26677 -- test samples 13323 -- K = 10
# ACC:  1.0
# [Finished in 112.7s]