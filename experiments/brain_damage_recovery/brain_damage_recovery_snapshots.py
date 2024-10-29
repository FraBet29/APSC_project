import os
import time
import argparse
import numpy as np
from dlroms import *
from geogp import GaussianRandomField
from fenics import Function, TestFunction
from fenics import solve, inner, grad, dx
import gmsh


if __name__ == '__main__':

	gmsh.initialize()

	parser = argparse.ArgumentParser(description="Generate snapshots for brain damage recovery example.")

	parser.add_argument('--num_snapshots', type=int, required=True, help="Number of snapshots to generate.")
	parser.add_argument('--train_test_split', type=float, default=0.9, help="Fraction of training snapshots.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for snapshots.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

	if args.train_test_split < 0 or args.train_test_split > 1:
		raise ValueError("Invalid fraction of training snapshots provided.")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Domain definition

	loop = lambda v: np.concatenate((v, v[[0]]))
	brain = np.load(os.path.join('brain_meshes', 'brainshape.npz'))
	domain = fe.polygon(loop(brain['main'][::9])) - fe.polygon(loop(brain['hole1'][::9])) - fe.polygon(loop(brain['hole2'][::8]))

	# Mesh and function space definition

	mesh_H = fe.loadmesh(os.path.join('brain_meshes', 'brain-mesh40.xml'))
	Vh_H = fe.space(mesh_H, 'CG', 1)
	h_H, nh_H = mesh_H.hmax(), Vh_H.dim()
	h_H, nh_H = mesh_H.hmax(), Vh_H.dim()

	# Random field generation

	print("Generating random field...")
	start = time.time()
	kernel = lambda r: np.exp(-100 * r ** 2)
	G_H = GaussianRandomField(domain, mesh_H, kernel=kernel, upto=30)
	u0_H = lambda seed: 0.5 * np.tanh(10 * (G_H.sample(seed) + 0.5)) + 0.5
	print(f"Random field generated. Elapsed time: {time.time() - start:.0f}s")

	# Sample a pair (mu, u) = (initial condition, time-to-recovery map)

	# Time iterator
	def step(w):

		D = 0.1 # diffusion coefficient
		r = 1000 # reaction coefficient
		dt = 0.0001 # time step size
		space = Vh_H

		Z, V = Function(space), TestFunction(space)
		W = fe.asvector(w, space)
		L = inner(Z, V) * dx  - inner(W, V) * dx + dt * D * inner(grad(Z), grad(V)) * dx # Z = u^{n+1}, W = u^n
		L = L - dt * r * inner(Z * (1 - Z), V) * dx
		Z.vector()[:] = w
		solve(L == 0, Z)

		return np.clip(Z.vector()[:], 0, 1)

	# Sampler for FKPP model
	def FKPP_sampler(seed):

		np.random.seed(seed)

		nt = 200 # number of time steps
		nh, u0 = nh_H, u0_H
		
		mu = u0(seed) # initial condition
		u = np.zeros((nt, nh))
		u[0] = u0(seed)

		for i in range(nt-1):
			u[i+1] = step(u[i])
		if np.any(u.max(axis=0) < 0.9):
			return FKPP_sampler(seed + 50000) # if the brain has not fully recovered, we try another simulation
		u = np.argmax(u >= 0.9, axis=0) / nt

		return mu, u

	# Snapshot sampling (via the function dlroms.roms.snapshots)

	print("Generating snapshots...")

	N_train = int(args.train_test_split * args.num_snapshots)

	snapshots(n=N_train, sampler=FKPP_sampler, verbose=args.verbose,
				filename=os.path.join(args.output_dir, 'snapshots_train.npz'))
	
	FKPP_sampler_test = lambda seed: FKPP_sampler(seed + 1000) # change the seed to generate different snapshots

	N_test = args.num_snapshots - N_train
	snapshots(n=N_test, sampler=FKPP_sampler_test, verbose=args.verbose,
				filename=os.path.join(args.output_dir, 'snapshots_test.npz'))
