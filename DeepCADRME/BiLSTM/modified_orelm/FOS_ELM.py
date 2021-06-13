
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
import pickle
"""
Paper: Online recurrent extreme learning machine and its application to time-series prediction
Code: https://github.com/chickenbestlover/Online-Recurrent-Extreme-Learning-Machine
"""

def orthogonalization(Arr):
  [Q, S, _] = np.linalg.svd(Arr)
  tol = max(Arr.shape) * np.spacing(max(S))
  r = np.sum(S > tol)
  Q = Q[:, :r]

  return Q

def linear(features, weights, bias):
  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights))
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]

  return V

def kernel(features, param):
  assert(features.shape[1] == weights.shape[1])
  
  (numSamples, numInputs) = features.shape
  omega_train = kernel_matrix(features)

  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights))
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]

  return V


def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H



class FOSELM(object):
  def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, LN=False, forgettingFactor=0.999, ORTH = False,RLS=False):

    self.activationFunction = activationFunction
    self.inputs = inputs
    self.outputs = outputs
    self.numHiddenNeurons = numHiddenNeurons

    # input to hidden weights
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.ORTH = ORTH

    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))
    self.LN = LN
    # auxiliary matrix used for sequential learning
    self.M = None
    self.forgettingFactor = forgettingFactor
    self.RLS=RLS

  def layerNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H - H.mean()) / (np.sqrt(H.var() + 0.0001))
    H_normalized = scaleFactor * H_normalized + biasFactor

    return H_normalized

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """
    if self.activationFunction is "sig":
      V = linear(features, self.inputWeights,self.bias)
      if self.LN:
        V = self.layerNormalization(V)
      H = sigmoidActFunc(V)
    else:
      print (" Unknown activation function type")
      raise NotImplementedError

    return H

  def _kernel_matrix(self, training_patterns, test_patterns=None):
    if test_patterns is None:
      omega = np.dot(training_patterns, training_patterns.conj().T)
    else:
      omega = np.dot(training_patterns, test_patterns.conj().T)
    return omega

  def save_weights(self, path):
    weights = {
    	'inputWeights': self.inputWeights,
    	'bias': self.bias}
    with open(path, 'wb') as f:
    	pickle.dump(weights, f)

  def load_weights(self, path):
    with open(path, 'rb') as f:
    	weights = pickle.load(f)
    	self.inputAE = weights['inputWeights']
    	self.bias = weights['bias']


  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    """
    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1
    #self.load_weights("model1.pkl")

    if self.ORTH:
      if self.numHiddenNeurons > self.inputs:
        self.inputWeights = orthogonalization(self.inputWeights)
      else:
        self.inputWeights = orthogonalization(self.inputWeights.transpose())
        self.inputWeights = self.inputWeights.transpose()

    if self.activationFunction is "sig":
      self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    else:
      print (" Unknown activation function type")
      raise NotImplementedError

    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])
    #file = 'weight/weightAE_'+str(self.num)+'.pkl'
    #self.save_weights(file)



  def train(self, features, targets):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
	
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)
    Ht = np.transpose(H)

    if self.RLS:

      self.RLS_k = np.dot(np.dot(self.M,Ht),inv( self.forgettingFactor*np.eye(numSamples)+ np.dot(H,np.dot(self.M,Ht))))
      self.RLS_e = targets - np.dot(H,self.beta)
      self.beta = self.beta + np.dot(self.RLS_k,self.RLS_e)
      self.M = 1/(self.forgettingFactor)*(self.M - np.dot(self.RLS_k,np.dot(H,self.M)))

    else:
      self.M = self.M - np.dot(self.M,
                                       np.dot(Ht, np.dot(
                                         pinv(np.eye(numSamples) + np.dot(H, np.dot(self.M, Ht))),
                                         np.dot(H, self.M))))
      self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))
      #self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      #self.beta = (self.forgettingFactor)*self.beta + (self.forgettingFactor)*np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))

  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    H = self.calculateHiddenLayerActivation(features)
    prediction = np.dot(H, self.beta)
    return prediction

