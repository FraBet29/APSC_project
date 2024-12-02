import unittest


class TestChannel(unittest.TestCase):
	
	def test_channel(self):
		from dlroms_bayesian.expansions import Channel
		import torch

		channel = Channel(0)
		x = torch.rand(10, 3, 5)
		y = channel(x)
		self.assertEqual(y.shape, (10, 5))
		self.assertTrue(torch.allclose(x[:, 0], y))


class TestDeterministic(unittest.TestCase):
	
	def test_deterministic(self):
		from dlroms.fespaces import unitsquaremesh, space
		from dlroms_bayesian.expansions import ExpandedLocal
		import torch

		mesh1 = unitsquaremesh(1, 1) # square with 4 nodes
		mesh2 = unitsquaremesh(2, 2) # square with 9 nodes
		space1 = space(mesh1, 'CG', 1)
		space2 = space(mesh2, 'CG', 1)
		layer = ExpandedLocal(space1, space2, support=0.7)
		layer.deterministic()
		self.assertTrue(layer.W().shape == (4, 9))
		# The top-left node in mesh2 has one neighbor in mesh1 (top-left), with distance 0
		self.assertAlmostEqual(layer.W()[0, 0].item(), 1.0)
		# The top-center node in mesh2 has two neighbors in mesh1 (top-left and top-right), with distance 0.5, hence weights 0.5 (normalized)
		self.assertAlmostEqual(layer.W()[0, 1].item(), 0.5)
		self.assertAlmostEqual(layer.W()[1, 1].item(), 0.5)


if __name__ == '__main__':
	unittest.main()
