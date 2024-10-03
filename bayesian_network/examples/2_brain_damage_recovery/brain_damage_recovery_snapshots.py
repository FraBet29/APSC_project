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
	parser.add_argument('--mode', type=str, choices=['h2h', 'h2c', 'c2c'], required=True, help="Mode of snapshot generation.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for snapshots.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Domain definition

	loop = lambda v: np.concatenate((v, v[[0]]))
	brain = np.load(os.path.join('brain_meshes', 'brainshape.npz'))
	domain = fe.polygon(loop(brain['main'][::9])) - fe.polygon(loop(brain['hole1'][::9])) - fe.polygon(loop(brain['hole2'][::8]))

	# Mesh and function space definition

	if args.mode == 'h2h' or args.mode == 'h2c':
		mesh_H = fe.mesh(domain, stepsize=0.01)
		Vh_H = fe.space(mesh_H, 'CG', 1)
		h_H, nh_H = mesh_H.hmax(), Vh_H.dim()

		# if args.verbose:
		# 	print(f"Stepsize of fine mesh: {h_H:.3f}")
		# 	print(f"Dimension of high-fidelity space: {nh_H}")

	if args.mode == 'c2c' or args.mode == 'h2c':
		mesh_C = fe.loadmesh(os.path.join('brain_meshes', 'brain-mesh30.xml'))
		Vh_C = fe.space(mesh_C, 'CG', 1)
		h_C, nh_C = mesh_C.hmax(), Vh_C.dim()

		# if args.verbose:
		# 	print(f"Stepsize of coarse mesh: {h_C:.3f}")
		# 	print(f"Dimension of low-fidelity space: {nh_C}")

	# Random field generation

	print("Generating random fields...")

	kernel = lambda r: np.exp(-100 * r ** 2)

	if args.mode == 'h2h' or args.mode == 'h2c':
		start = time.time()
		G_H = GaussianRandomField(domain, mesh_H, kernel=kernel, upto=30)
		u0_H = lambda seed: 0.5 * np.tanh(10 * (G_H.sample(seed) + 0.5)) + 0.5
		print(f"High-fidelity random field, elapsed time: {time.time() - start:.2f}s")

	if args.mode == 'c2c' or args.mode == 'h2c':
		start = time.time()
		G_C = GaussianRandomField(domain, mesh_C, kernel=kernel, upto=30)
		u0_C = lambda seed: 0.5 * np.tanh(10 * (G_C.sample(seed) + 0.5)) + 0.5
		print(f"Low-fidelity random field, elapsed time: {time.time() - start:.2f}s")

	# Sample a pair (mu, u) = (initial condition, time-to-recovery map)

	# Time iterator
	def step(w):

		D = 0.1 # diffusion coefficient
		r = 1000 # reaction coefficient
		dt = 0.0001 # time step size

		if args.mode == 'c2c' or args.mode == 'h2c':
			space = Vh_C
		elif args.mode == 'h2h':
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

		if args.mode == 'c2c':
			nh, u0 = nh_C, u0_C
		elif args.mode == 'h2h' or args.mode == 'h2c':
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

	N_train = int(0.9 * args.num_snapshots)
	snapshots(n=N_train, sampler=FKPP_sampler, verbose=args.verbose,
				filename=os.path.join(args.output_dir, 'snapshots_train_' + args.mode + '.npz'))
	
	FKPP_sampler_test = lambda seed: FKPP_sampler(seed + 1000)
	N_test = args.num_snapshots - N_train
	snapshots(n=N_test, sampler=FKPP_sampler_test, verbose=args.verbose,
				filename=os.path.join(args.output_dir, 'snapshots_test_' + args.mode + '.npz'))
