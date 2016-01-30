import struct
import math
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy import stats

class NeuralNet:
	def __init__(self,train_data,train_label,test_data,test_label,hidden_size,train_size,test_size,learning_rate,momentum,regularize,entropy_diff,gradCheckEpsilon,gradChecking,activation):
		self.a2 = []
		self.del2 = []
		self.z2 =[]
		self.train_size = train_size
		self.test_size = test_size
		self.it = 0
		self.tm2 = 0
		self.activation = activation
		self.gradChecking = gradChecking
		self.gradCheckEpsilon = gradCheckEpsilon
		self.momentum = momentum
		self.entropy_diff = entropy_diff
		self.regularize = regularize
		self.learning_rate = learning_rate
		self.hidden_size = hidden_size
		self.Y_train = np.zeros((self.train_size,10))
		self.Y_test = np.zeros((self.test_size,10))
		self.theta1 = np.random.rand(785,self.hidden_size)
		self.theta2 = np.random.rand(self.hidden_size+1,10)
		self.theta1_tm2 = np.copy(self.theta1)
		self.theta2_tm2 = np.copy(self.theta2)
		self.velocity_theta1 = np.zeros((785,self.hidden_size))
		self.velocity_theta2 = np.zeros((self.hidden_size+1,10))
		self.grad1 = np.zeros((785,self.hidden_size))
		self.grad2 = np.zeros((self.hidden_size+1,10))
		self.iteration = 0
		with open(train_label, "r") as flbl:
			magic, num = struct.unpack(">II", flbl.read(8))
			self.y_train = np.fromfile(flbl, dtype=np.int8)

		with open(train_data, "r") as fimg:
			magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
			self.X_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.y_train), rows, cols)
			self.X_train = np.array([ np.hstack(self.X_train[i]) for i in range(self.train_size)])
			self.X_train = np.matrix(np.hstack((np.ones((self.X_train.shape[0], 1), dtype=self.X_train.dtype),self.X_train)))
			self.y_train = np.matrix(self.y_train[0:self.train_size])

		with open(test_label, "r") as flbl:
			magic, num = struct.unpack(">II", flbl.read(8))
			self.y_test = np.fromfile(flbl, dtype=np.int8)

		with open(test_data, "r") as fimg:
			magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
			self.X_test = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.y_test), rows, cols)
			self.X_test = np.array([ np.hstack(self.X_test[i]) for i in range(self.test_size)])
			self.X_test = np.matrix(np.hstack((np.ones((self.X_test.shape[0], 1), dtype=self.X_test.dtype),self.X_test)))
			self.y_test = np.matrix(self.y_test[0:self.test_size])
		
		for i in range(self.train_size):
			self.Y_train[i,self.y_train[0,i]] = 1

		for i in range(self.test_size):
			self.Y_test[i,self.y_test[0,i]] = 1 

		self.findZScores()

	def findZScores(self):
		self.Z_train = stats.zscore(self.X_train,axis = 1)
		self.Z_test = stats.zscore(self.X_test,axis = 1)

	def sigmoid(self,mat):
		return np.matrix(1/(np.exp(-mat)+1))

	def tanh(self,mat):
		return np.tanh(mat)

	def relu(self,mat):
		return np.maximum(mat,0)

	def softmax(self,data):
		e = np.exp(data)
		return e/np.sum(e,axis=1, keepdims=True)

	def forwardPropogationStochastic(self):
		self.a2 = np.matrix(self.Z_train[self.iteration,:]) * np.matrix(self.theta1)
		if self.activation=="sigmoid":
			self.z2 = self.sigmoid(self.a2)
		if self.activation=="tanh":
			self.z2 = self.tanh(self.a2)
		if self.activation=="relu":
			self.z2 = self.relu(self.a2)
		bias = np.zeros((1,1))
		self.z2 = np.concatenate((bias,self.z2),axis = 1)
		self.a3 = np.matrix(self.z2) * np.matrix(self.theta2)
		self.z3 = self.softmax(self.a3)
		return self.z3

	def forwardPropogationBatch(self,dataset="train"):
		self.z2 = []
		b = 0
		if dataset == "train":
			self.z2 = np.matrix(self.Z_train)*np.matrix(self.theta1)
			b = self.train_size
		else:
			self.z2 = np.matrix(self.Z_test)*np.matrix(self.theta1)
			b = self.test_size
		if self.activation=="sigmoid":
			self.a2 = self.sigmoid(self.z2)
		if self.activation=="tanh":
			self.a2 = self.tanh(self.z2)
		if self.activation=="relu":
			self.a2 = self.relu(self.z2)
		bias = np.ones((b,1))
		self.a2 = np.concatenate((bias,self.a2),axis = 1)
		self.z3 = np.matrix(self.a2)*np.matrix(self.theta2)
		self.a3 = self.softmax(self.z3)
		return self.a3

	def backPropogationStochastic(self):
		if self.it >= 2:
			self.velocity_theta1 = self.theta1 - self.theta1_tm2
			self.velocity_theta2 = self.theta2 - self.theta2_tm2
			self.theta2_tm2 = np.copy(self.theta2)
			self.theta1_tm2 = np.copy(self.theta1)
		y = self.forwardPropogationStochastic()
		self.del3 = self.Y_train[self.iteration] - y
		if self.activation=="sigmoid":
			self.del2 = np.multiply(np.transpose(self.theta2 * np.transpose(self.del3)) , np.multiply(self.z2,1-self.z2))
		if self.activation=="tanh":
			self.del2 = np.multiply(np.transpose(self.theta2 * np.transpose(self.del3)) , 1-np.multiply(self.z2,self.z2))
		if self.activation=="relu":
			self.del2 = np.multiply(np.transpose(self.theta2 * np.transpose(self.del3)) , (self.z2>0).astype(int))
		self.grad2 = np.transpose(self.z2) * self.del3
		self.grad1 = (np.reshape(self.Z_train[self.iteration,:],(785,1)) * self.del2)[:,1:]
		if self.gradChecking:
			gradCheck_theta1 = np.zeros((785,self.hidden_size))
			gradCheck_theta2 = np.zeros((self.hidden_size+1,10))
			theta1_temp = np.copy(self.theta1)
			theta2_temp = np.copy(self.theta2)
			for i in range(785):
				for j in range(self.hidden_size):
					self.theta1[i,j] += self.gradCheckEpsilon
					e1 = self.entropy()
					self.theta1[i,j] -= (2 * self.gradCheckEpsilon)
					e2 = self.entropy()
					gradCheck_theta1[i,j] = float(e1-e2)/(2*self.gradCheckEpsilon) 
					self.theta1[i,j] += (2 * self.gradCheckEpsilon)
			for i in range(self.hidden_size+1):
				for j in range(10):
					self.theta2[i,j] += self.gradCheckEpsilon
					e1 = self.entropy()
					self.theta2[i,j] -= (2 * self.gradCheckEpsilon)
					e2 = self.entropy()
					gradCheck_theta2[i,j] = float(e1-e2)/(2*self.gradCheckEpsilon) 
					self.theta2[i,j] += (2 * self.gradCheckEpsilon)
			self.theta1 = np.copy(theta1_temp)
			self.theta2 = np.copy(theta2_temp)
			print "grad check thet1= " + str(abs(np.sum(gradCheck_theta1 - self.grad1))/(785*self.hidden_size))
			print "grad check thet2= " + str(abs(np.sum(gradCheck_theta2 - self.grad2))/(10*(self.hidden_size+1)))
		self.theta2 = self.theta2 + (((self.grad2) - (self.regularize * self.theta2)) * self.learning_rate) + (self.momentum * self.velocity_theta2)
		self.theta1 = self.theta1 + ((self.grad1 - (self.regularize * self.theta1)) * self.learning_rate) + (self.momentum * self.velocity_theta1)
		self.iteration += 1
		if self.iteration == self.train_size:
			self.iteration = 0

	def findAccuracy(self,dataset = "train"):
		fp = self.forwardPropogationBatch(dataset)
		op = np.argmax(fp,axis = 1)
		if dataset == "train":
			acc = np.sum((op == np.transpose(self.y_train)).astype(int))
			return (float(acc)/self.train_size)*100
		else:
			acc = np.sum((op == np.transpose(self.y_test)).astype(int))
			return (float(acc)/self.test_size)*100

	def entropy(self):
		fp = self.forwardPropogationBatch()
		fp_log = np.log(fp)
		fp_log[fp_log == -inf] = 0
		return np.sum(np.multiply(fp_log,self.Y_train))

	def run(self):
		ent = self.entropy()
		xaxis = []
		yaxis_train = []
		yaxis_test = []
		while True:
			self.backPropogationStochastic()
			if self.it % 5000 == 0:
				xaxis.append(self.it)
				yaxis_test.append(self.findAccuracy("test"))
				yaxis_train.append(self.findAccuracy("train"))
			self.it += 1
			if self.iteration == self.train_size-1:
				acc = self.findAccuracy()
				print "accuracy = " + str(acc)
				ent_cur = self.entropy()
				ent_diff = abs(float(abs(ent) - abs(ent_cur)))
				print "entropy diff = " + str(ent_diff)
				if ent_diff <= self.entropy_diff:
					break
				ent = ent_cur
		print self.findAccuracy("test")
		plt.plot(xaxis,yaxis_train,'b-')
		plt.plot(xaxis,yaxis_test,'r-')
		plt.xlabel("Iterations")
		plt.ylabel("Accuracy")
		plt.title("hiddenUnits:100,eta:0.001,lamda:0,momentum:0,activation:sigmoid")
		plt.show()

def main():
	nn = NeuralNet("train-images.idx3-ubyte","train-labels.idx1-ubyte","t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte",100,60000,10000,0.001,0,0,450,0.00001,False,"sigmoid")
	nn.run()

if __name__ == '__main__':
	main()
