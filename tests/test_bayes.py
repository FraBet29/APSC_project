import unittest


class TestVariationalInference(unittest.TestCase):
	
	def test_abstract_raise_typeerror(self):
		from dlroms.roms import DFNN
		from dlroms_bayesian.bayesian import Bayesian, VariationalInference

		model = DFNN()
		bayes = Bayesian(model)
		self.assertRaises(TypeError, VariationalInference, bayes)


class TestLogLikelihood(unittest.TestCase):

	def test_gaussian_log_likelihood(self):
			from dlroms_bayesian.bayesian import gaussian_log_likelihood
			import torch

			target = torch.tensor([1.0, 2.0, 3.0])
			output = torch.tensor([1.0, 2.0, 3.0])
			log_beta = torch.tensor([1.0])
			ntrain = 1
			self.assertEqual(gaussian_log_likelihood(target, output, log_beta, ntrain), torch.tensor([0.5]))


class TestBayesian(unittest.TestCase):

	def test_bayesian_init(self):
		from dlroms_bayesian.bayesian import Bayesian
		from dlroms.roms import DFNN

		model = DFNN()
		bayes = Bayesian(model)
		self.assertEqual(bayes.model, model)

	def test_bayesian_init_raise_typeerror(self):
		from dlroms_bayesian.bayesian import Bayesian
		import torch

		self.assertRaises(TypeError, Bayesian, torch.tensor([1.0])) # not a ROM model

	def test_bayesian_no_trainer_raise_valueerror(self):
		from dlroms_bayesian.bayesian import Bayesian
		from dlroms.roms import DFNN
		import torch

		model = DFNN()
		bayes = Bayesian(model) # trainer is None
		self.assertRaises(ValueError, bayes.forward, torch.tensor([1.0]))
		self.assertRaises(ValueError, bayes.train, torch.tensor([1.0]), torch.tensor([1.0]), 1, 1, None, 1.0, 1.0)
		self.assertRaises(ValueError, bayes.sample, torch.tensor([1.0]), 1)


if __name__ == '__main__':
	unittest.main()