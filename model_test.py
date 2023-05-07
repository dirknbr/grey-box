from model import SemiLinear
import unittest
import numpy as np

class TestSemiLinear(unittest.TestCase):
  def test_model(self):
  	n, k = 100, 3
  	np.random.seed(22)
  	dtype = np.float32
  	X = np.random.normal(0, 1, (n, k)).astype(dtype)
  	Z = np.random.normal(0, 1, (n, 2)).astype(dtype)
  	y = 1 + X.dot([1, -1, 1]) + Z[:, 0] + np.random.normal(0, 1, n)
  	y = y.astype(dtype)
  	# print(X.dtype, Z.dtype, y.dtype)
  	model = SemiLinear(numx=k)
  	model.fit(X, Z, y, epochs=20)
  	pred = model.predict(X, Z)
  	# check that Z has an effect
  	pred_cf = model.predict(X, Z - 1) # counterfactual

  	print('beta', model.get_beta())
  	print('mean', y.mean(), np.mean(pred), np.mean(pred_cf))

  	self.assertEqual(len(y), pred.shape[0])
  	self.assertEqual(len(model.get_beta()), k)
  	self.assertTrue(np.mean(pred) > np.mean(pred_cf))


if __name__ == '__main__':
  unittest.main()  	