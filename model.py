import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
  def __init__(self, hidden, numx=1):
  	super().__init__()
  	self._layers = [] 	
  	for hi in hidden:
  	  self._layers.append(tf.keras.layers.Dense(hi))
  	self._layers.append(tf.keras.layers.Dense(1))
  	self.beta = tf.Variable(np.random.normal(0, 1, (numx, 1)).astype(np.float32))

  def call(self, inputs):
  	X, Z = inputs
  	for layer in self._layers:
  	  Z = layer(Z)
    #TODO: add classification
  	return X @ self.beta + Z


class SemiLinear():
  def __init__(self, method='regression', numx=1, hidden=[2], 
  	           loss='mse', lr=.01):
    """Semi Linear model y = X @ beta + NN(Z).
    Args:
      method: regression or classification
      numx: number of linear variables
      hidden: list of hidden nodes (Z)
      loss: loss function string
      lr: learning rate
    """
    self.method = method
    self.hidden = hidden
    self.loss = loss
    self.lr = lr
    self.numx = numx
    self.model = Model(hidden, numx)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.model.compile(loss=loss, optimizer=self.optimizer)


  def fit(self, X, Z, y, epochs=20):
  	"""Fits the model.
  	Args:
  	  X: matrix of linear features
  	  Z: matrix of nonlinear features
  	  y: vector of target variable
  	"""
  	assert self.numx == X.shape[1]
  	self.model.fit([X, Z], y, epochs=epochs)

  def predict(self, X, Z):
  	return self.model.predict([X, Z])

  def get_beta(self):
  	return self.model.beta.numpy().flatten()

