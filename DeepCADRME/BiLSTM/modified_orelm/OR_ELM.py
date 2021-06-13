import numpy as np
import math
from numpy.linalg import pinv
from numpy.linalg import inv
from DeepCADRME.BiLSTM.modified_orelm.FOS_ELM import FOSELM

"""
Paper: Online recurrent extreme learning machine and its application to time-series prediction
Code: https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine
"""

def orthogonalization(Arr):
	[Q, S, _] = np.linalg.svd(Arr)
	tol = max(Arr.shape) * np.spacing(max(S))
	r = np.sum(S > tol)
	Q = Q[:, :r]


def sigmoidActFunc(features, weights, bias):
	assert(features.shape[1] == weights.shape[1])
	(numSamples, numInputs) = features.shape
	(numHiddenNeuron, numInputs) = weights.shape
	V = np.dot(features, np.transpose(weights))
	for i in range(numHiddenNeuron):
		V[:, i] += bias[0, i]
	H = 1 / (1+np.exp(-V))
	return H


def linear_recurrent(features, inputW,hiddenW,hiddenA, bias):
	(numSamples, numInputs) = features.shape
	(numHiddenNeuron, numInputs) = inputW.shape
	V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA,hiddenW)
	for i in range(numHiddenNeuron):
		V[:, i] += bias[0, i]

	return V

def sigmoidAct_forRecurrent(features,inputW,hiddenW,hiddenA,bias):
	(numSamples, numInputs) = features.shape
	(numHiddenNeuron, numInputs) = inputW.shape
	V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA,hiddenW)
	for i in range(numHiddenNeuron):
		V[:, i] += bias[0, i]
	H = 1 / (1 + np.exp(-V))
	return H

def sigmoidActFunc(V):
	H = 1 / (1+np.exp(-V))
	#H = np.tanh(V)
	return H


class ORELM:
	def __init__(self, inputs, numHiddenNeurons, activationFunction, outputs=0, LN=True, AE=True, ORTH=True):

		self.activationFunction = activationFunction
		self.inputs = inputs
		self.outputs = outputs
		self.numHiddenNeurons = numHiddenNeurons

		# input to hidden weights
		self.inputWeights  = np.random.random((self.numHiddenNeurons, self.inputs))
		# hidden layer to hidden layer wieghts
		self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
		# initial hidden layer activation
		self.initial_H = np.random.random((1, self.numHiddenNeurons)) * 2 -1
		self.H = self.initial_H
		self.LN = LN
		self.AE = AE
		self.ORTH = ORTH
		# bias of hidden units
		self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
		# hidden to output layer connection
		self.beta = np.random.random((self.numHiddenNeurons, self.outputs))

		# auxiliary matrix used for sequential learning
		self.M = inv(0.00001 * np.eye(self.numHiddenNeurons))

		#self.forgettingFactor = outputWeightForgettingFactor

		self.trace=0
		self.thresReset=0.001
    
		self.numSampleInClass1 = 0
		self.numSampleInClass2 = 0
		self.numSampleInClass3 = 0
		self.num = 0



		if self.AE:
			self.inputAE = FOSELM(inputs = inputs,
                            outputs = inputs,
                            numHiddenNeurons = numHiddenNeurons,
                            activationFunction = activationFunction,
                            LN= LN,
                            ORTH = ORTH
                            )

			self.hiddenAE = FOSELM(inputs = numHiddenNeurons,
                             outputs = numHiddenNeurons,
                             numHiddenNeurons = numHiddenNeurons,
                             activationFunction=activationFunction,
                             LN= LN,
                             ORTH = ORTH
                             )
  
	def layerNormalization(self, H, scaleFactor=1, biasFactor=0):

		H_normalized = (H-H.mean())/(np.sqrt(H.var() + 0.000001))
		H_normalized = scaleFactor*H_normalized+biasFactor

		return H_normalized
     
	def __softmax(self, x):
		c = np.max(x, axis=1).reshape(-1, 1)
		upper = np.exp(x - c)
		lower = np.sum(upper, axis=1).reshape(-1, 1)
		return upper / lower

	def __calculateInputWeightsUsingAE(self, features):
		self.inputAE.train(features=features,targets=features)
		return self.inputAE.beta

	def __calculateHiddenWeightsUsingAE(self, features):
		self.hiddenAE.train(features=features,targets=features)
		return self.hiddenAE.beta

	def calculateHiddenLayerActivation(self, features):
		"""
		Calculate activation level of the hidden layer
		:param features feature matrix with dimension (numSamples, numInputs)
		:return: activation level (numSamples, numHiddenNeurons)
		"""
		if self.activationFunction is "sig":

			if self.AE:
				self.inputWeights = self.__calculateInputWeightsUsingAE(features)

				self.hiddenWeights = self.__calculateHiddenWeightsUsingAE(self.H)

			V = linear_recurrent(features=features,
                           inputW=self.inputWeights,
                           hiddenW=self.hiddenWeights,
                           hiddenA=self.H,
                           bias= self.bias)
			if self.LN:
				V = self.layerNormalization(V)
			self.H = sigmoidActFunc(V)

		else:
			print (" Unknown activation function type")
			raise NotImplementedError
		return self.H


	def initializePhase(self, lamb=0.0001):
		"""
		Step 1: Initialization phase
		:param features feature matrix with dimension (numSamples, numInputs)
		:param targets target matrix with dimension (numSamples, numOutputs)
		"""



		if self.activationFunction is "sig":
			self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
      #self.load_weights("model.pkl")
		else:
			print (" Unknown activation function type")
			raise NotImplementedError

		self.M = inv(lamb*np.eye(self.numHiddenNeurons))
		self.beta = np.zeros([self.numHiddenNeurons,self.outputs])
    

		# randomly initialize the input->hidden connections
		self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
		self.inputWeights = self.inputWeights * 2 - 1

		if self.AE:
			self.inputAE.initializePhase(lamb=0.00001)
			self.hiddenAE.initializePhase(lamb=0.00001)
		else:
			# randomly initialize the input->hidden connections
			self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
			self.inputWeights = self.inputWeights * 2 - 1

			if self.ORTH:
				if self.numHiddenNeurons > self.inputs:
					self.inputWeights = orthogonalization(self.inputWeights)
				else:
					self.inputWeights = orthogonalization(self.inputWeights.transpose())
					self.inputWeights = self.inputWeights.transpose()

			# hidden layer to hidden layer wieghts
			self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
			self.hiddenWeights = self.hiddenWeights * 2 - 1
			if self.ORTH:
				self.hiddenWeights = orthogonalization(self.hiddenWeights)
 

	def reset(self):
		self.H = self.initial_H


	def predict(self, features):
		"""
		Make prediction with feature matrix
		:param features: feature matrix with dimension (numSamples, numInputs)
		:return: predictions with dimension (numSamples, numOutputs)
		"""
		H = self.calculateHiddenLayerActivation(features)
		prediction = np.dot(H, self.beta)
		return prediction

	def predict_H(self, features):
		H = self.calculateHiddenLayerActivation(features)
		return H

