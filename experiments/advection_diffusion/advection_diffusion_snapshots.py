import os
import time
import argparse
import numpy as np
from dlroms import *
from dolfin import *


def sampler(seed, V):
	"""
	Generate N pairs (mu, u) of parameters and solutions in the finite element space V.
	"""
	np.random.seed(seed)
	mu = np.random.uniform(0., 1., size=(4,))
	mu1, mu2, mu3, mu4 = mu

	u_D = Expression('0.01', degree=0)

	def boundary(x, on_boundary):
		return on_boundary

	bc = DirichletBC(V, u_D, boundary) # Dirichlet boundary conditions

	f = Expression('100 * (x[0] * x[1] - x[1] * x[1])', degree=2) # source term

	sigma = Expression('6 + 5 * tanh(20 * (x[1] + 10 * mu1 * x[0] * (x[0] - 1) * (x[0] - mu2) * (x[0] - mu3) - 0.5))',
						degree=2, mu1=mu1, mu2=mu2, mu3=mu3) # diffusion coefficient

	beta = Expression(('10 * cos(2 * pi * mu4)', '10 * sin(2 * pi * mu4)'),
						degree=2, mu4=mu4, pi=np.pi) # advection coefficient

	u = TrialFunction(V)
	v = TestFunction(V)

	F = sigma * dot(grad(u), grad(v)) * dx + dot(beta, grad(u)) * v * dx - f * v * dx
	a, L = lhs(F), rhs(F)

	u = Function(V)
	solve(a == L, u, bc)

	return mu, u.vector().get_local()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Generate snapshots for model problem with autoencoder example.")

	parser.add_argument('--num_snapshots', type=int, help="Number of snapshots to generate.")
	parser.add_argument('--multi_fidelity', action='store_true', help="Generate multi-fidelity snapshots.")
	parser.add_argument('--num_snapshots_high', type=int, help="Number of high-fidelity snapshots to generate.")
	parser.add_argument('--num_snapshots_low', type=int, help="Number of low-fidelity snapshots to generate.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for snapshots.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

	if (not args.multi_fidelity) and (args.num_snapshots is None):
		raise ValueError("Number of snapshots must be specified.")

	if args.multi_fidelity and (args.num_snapshots_high is None or args.num_snapshots_low is None):
		raise ValueError("Number of high-fidelity and low-fidelity snapshots must be specified.")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Domain and mesh definition

	mesh_H = fe.unitsquaremesh(100, 100) # fine mesh
	V_H = fe.space(mesh_H, 'CG', 1) 
	Nh_H = V_H.dim()

	if args.multi_fidelity:
		mesh_C = fe.unitsquaremesh(50, 50) # coarse mesh
		V_C = fe.space(mesh_C, 'CG', 1)
		Nh_C = V_C.dim()

	# Snapshot sampling (via the function dlroms.roms.snapshots)

	print("Generating snapshots...")

	if args.multi_fidelity:
		sampler_high = lambda seed: sampler(seed, V_H)
		snapshots(n=args.num_snapshots_high, sampler=sampler_high, verbose=args.verbose,
					filename=os.path.join(args.output_dir, 'snapshots_train_H_' + str(args.num_snapshots_high) + '.npz'))
		sampler_low = lambda seed: sampler(seed + 1000, V_C)
		snapshots(n=args.num_snapshots_low, sampler=sampler_low, verbose=args.verbose,
					filename=os.path.join(args.output_dir, 'snapshots_train_C_' + str(args.num_snapshots_low) + '.npz'))
	else:
		sampler_test = lambda seed: sampler(seed + 10000, V_H)
		snapshots(n=args.num_snapshots, sampler=sampler_test, verbose=args.verbose,
					filename=os.path.join(args.output_dir, 'snapshots_test.npz'))