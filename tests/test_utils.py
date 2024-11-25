import unittest


class TestMultiMse(unittest.TestCase):
	
	def test_multi_mse(self):
		from dlroms.roms import euclidean
		from dlroms_bayesian.utils import multi_mse
		import torch

		true = torch.zeros(10, 2, 3)
		pred = torch.ones(10, 2, 3)
		mse = multi_mse(euclidean)(true, pred)
		self.assertEqual(mse.shape, ())

	def test_multi_mse_reduce(self):
		from dlroms.roms import euclidean
		from dlroms_bayesian.utils import multi_mse
		import torch

		true = torch.zeros(10, 2, 3)
		pred = torch.ones(10, 2, 3)
		mse = multi_mse(euclidean)(true, pred, reduce=False)
		self.assertEqual(mse.shape, (2,))


class TestGaussian(unittest.TestCase):

	def test_log_prob_scalar(self):
		from dlroms_bayesian.utils import Gaussian
		import torch
		import math

		gaussian = Gaussian() # mu = 0, sigma = 1
		x = torch.tensor([0.])
		log_prob = gaussian.log_prob(x)
		self.assertEqual(log_prob.shape, ())
		self.assertAlmostEqual(log_prob.item(), math.log(1. / math.sqrt(2 * math.pi)))

	def test_log_prob_tensor(self):
		from dlroms_bayesian.utils import Gaussian
		import torch
		import math

		gaussian = Gaussian(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
		x = torch.tensor([0., 0.])
		log_prob = gaussian.log_prob(x)
		self.assertEqual(log_prob.shape, ())
		self.assertAlmostEqual(log_prob.item(), 2 * math.log(1. / math.sqrt(2 * math.pi)))


if __name__ == '__main__':
	unittest.main()