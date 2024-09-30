import os
import time
import argparse
import numpy as np
from dlroms import *
from dlroms.gp import GaussianRandomField
from fenics import FiniteElement, MixedElement, FunctionSpace
from fenics import Function, TrialFunctions, TestFunctions
from fenics import solve, dot, grad, dx
import gmsh


# Injection term
class Injection(UserExpression):
	
	def __init__(self, w, r, **kwargs):
		self.w = w
		self.r = r
		super().__init__(**kwargs)

	def eval(self, values, x):
		if abs(x[0] - 0.5 * self.w) <= 0.5 * self.w and abs(x[1] - 0.5 * self.w) <= 0.5 * self.w:
			values[0] = self.r
		elif abs(x[0] + 0.5 * self.w - 1) <= 0.5 * self.w and abs(x[1] + 0.5 * self.w - 1) <= 0.5 * self.w:
			values[0] = - self.r
		else:
			values[0] = 0

	def value_shape(self):
		return ()


if __name__ == '__main__':

	gmsh.initialize()

	parser = argparse.ArgumentParser(description="Generate snapshots for Darcy flow example.")

	parser.add_argument('--num_snapshots', type=int, required=True, help="Number of snapshots to generate.")
	# parser.add_argument('--mode', type=str, choices=['h2h', 'h2c', 'c2c'], required=True, help="Mode of snapshot generation.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for snapshots.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Domain and mesh definition

	domain = fe.rectangle((0.0, 0.0), (1.0, 1.0))
	mesh = fe.mesh(domain, stepsize=0.05)

	# Random field generation

	class Field(UserExpression):
		def __init__(self, field, **kwargs):
			self.field = field
			super().__init__(**kwargs)

		def eval(self, values, x):
			K = Function(Vh) # NOTE: K is a function in the same space as the random field
			K.vector()[:] = np.exp(self.field)
			values[0] = K(x)

		def value_shape(self):
			return ()

	l = 0.1 # length scale
	kernel = lambda r: np.exp(- np.abs(r) / l)
	G = GaussianRandomField(mesh, kernel=kernel, upto=50) # Euclidean version

	def Darcy_solver(seed):
		"""
		Solve the Darcy problem for a given instance of the input random field.
		"""
		r = 10 # source rate
		w = 0.125 # source size

		# Function spaces
		CG_elem = FiniteElement('CG', mesh.ufl_cell(), 1)
		R_elem = FiniteElement('R', mesh.ufl_cell(), 0) # real number for Lagrange multiplier
		# W_elem = MixedElement([CG_elem, R_elem]) # primal formulation (pressure)

		Wh = FunctionSpace(mesh, CG_elem * R_elem) # mixed function space

		# Trial and test functions
		p, mu = TrialFunctions(Wh)
		q, nu = TestFunctions(Wh)

		f = Injection(w, r, degree=0) # f only takes constant values

		# Gaussian random field as an expression
		K = Field(G.sample(seed), degree=1)

		# Variational problem
		a = K * dot(grad(p), grad(q)) * dx + nu * p * dx + mu * q * dx
		L = f * q * dx

		w = Function(Wh)
		solve(a == L, w)
		p, mu = w.split(True)

		# Compute u from p
		RT_elem = FiniteElement('RT', mesh.ufl_cell(), 3)
		Vh = FunctionSpace(mesh, RT_elem)
		u = Function(Vh)
		u.assign(project(- K * grad(p), Vh))

		return u, p
	
	print("Generating snapshots...")

	Vh = FunctionSpace(mesh, 'CG', 1) # used for K, p and the components of u

	K_data = np.zeros((args.num_snapshots, Vh.dim()))
	p_data = np.zeros((args.num_snapshots, Vh.dim()))
	u_x_data = np.zeros((args.num_snapshots, Vh.dim()))
	u_y_data = np.zeros((args.num_snapshots, Vh.dim()))

	timer_start = time.time()

	for n in range(args.num_snapshots):

		if n % 100 == 0:
			print(f'Generating snapshot {n+1} of {args.num_snapshots} (elapsed time: {time.time() - timer_start:.2f} s)')
		
		u, p = Darcy_solver(G, n)

		u_x = project(u[0], Vh)
		u_y = project(u[1], Vh)

		K_data[n] = np.exp(G.sample(n))
		p_data[n] = p.vector().get_local()
		u_x_data[n] = u_x.vector().get_local()
		u_y_data[n] = u_y.vector().get_local()

	N_train = int(0.9 * args.num_snapshots)
	np.save(os.path.join(args.output_dir, 'K_train.npy'), K_data[:N_train])
	np.save(os.path.join(args.output_dir, 'p_train.npy'), p_data[:N_train])
	np.save(os.path.join(args.output_dir, 'u_x_train.npy'), u_x_data[:N_train])
	np.save(os.path.join(args.output_dir, 'u_y_train.npy'), u_y_data[:N_train])

	np.save(os.path.join(args.output_dir, 'K_test.npy'), K_data[N_train:])
	np.save(os.path.join(args.output_dir, 'p_test.npy'), p_data[N_train:])
	np.save(os.path.join(args.output_dir, 'u_x_test.npy'), u_x_data[N_train:])
	np.save(os.path.join(args.output_dir, 'u_y_test.npy'), u_y_data[N_train:])