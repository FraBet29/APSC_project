import unittest


class TestParameterToVector(unittest.TestCase):
	
	def test_parameter_to_vector(self):
		from dlroms_bayesian.svgd import parameters_to_vector
		import torch

		model = torch.nn.Linear(2, 1)
		model.weight.data.fill_(1.)
		model.bias.data.fill_(0.)
		params = parameters_to_vector(model.parameters(), grad=False)
		self.assertTrue(torch.allclose(params, torch.tensor([1., 1., 0.])))


class TestVectorToParameters(unittest.TestCase):

	def test_vector_to_parameters(self):
		from dlroms_bayesian.svgd import vector_to_parameters
		import torch

		model = torch.nn.Linear(2, 1)
		model.weight.data.fill_(1.)
		model.bias.data.fill_(0.)
		params = torch.tensor([2., 2., 1.])
		vector_to_parameters(params, model.parameters(), grad=False)
		self.assertTrue(torch.allclose(model.weight, torch.tensor([[2., 2.]])))
		self.assertTrue(torch.allclose(model.bias, torch.tensor([1.])))


class TestRBFKernel(unittest.TestCase):

	def test_rbf_kernel(self):
		from dlroms_bayesian.svgd import RBFKernel
		import torch

		# TODO


class TestSVGD(unittest.TestCase):

	def test_call(self):
		from dlroms.roms import DFNN
		from dlroms_bayesian.bayesian import Bayesian
		from dlroms_bayesian.svgd import SVGD
		import torch

		layer = torch.nn.Linear(2, 1)
		layer.weight.data.fill_(1.)
		layer.bias.data.fill_(0.)
		model = DFNN(layer)
		bayes = Bayesian(model)
		svgd = SVGD(bayes, n_samples=3)
		bayes.set_trainer(svgd)
		x = torch.zeros(10, 2)
		y = bayes(x)
		self.assertEqual(y.shape, (10, 1))

	def test_call_no_reduce(self):
		from dlroms.roms import DFNN
		from dlroms_bayesian.bayesian import Bayesian
		from dlroms_bayesian.svgd import SVGD
		import torch

		layer = torch.nn.Linear(2, 1)
		layer.weight.data.fill_(1.)
		layer.bias.data.fill_(0.)
		model = DFNN(layer)
		bayes = Bayesian(model)
		svgd = SVGD(bayes, n_samples=3)
		bayes.set_trainer(svgd)
		x = torch.zeros(10, 2)
		y = bayes(x, reduce=False)
		self.assertEqual(y.shape, (3, 10, 1))

	def test_train_other_trainer(self):
		from dlroms.roms import DFNN
		from dlroms_bayesian.bayesian import Bayesian
		from dlroms_bayesian.svgd import SVGD
		import torch

		layer = torch.nn.Linear(2, 1)
		layer.weight.data.fill_(1.)
		layer.bias.data.fill_(0.)
		model = DFNN(layer)
		bayes = Bayesian(model)
		svgd1 = SVGD(bayes, n_samples=3)
		svgd2 = SVGD(bayes, n_samples=3)
		bayes.set_trainer(svgd1)
		x, y = torch.zeros(10, 2), torch.zeros(10, 1)
		self.assertRaises(RuntimeError, svgd2.train, x, y, 10, 1)


if __name__ == '__main__':
	unittest.main()