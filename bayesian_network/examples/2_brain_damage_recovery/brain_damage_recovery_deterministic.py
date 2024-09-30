import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from dlroms import *
import gmsh
import sys
sys.path.append(os.path.join("..", "..", "dlroms")) # TODO: better alternative?


if __name__ == '__main__':

	gmsh.initialize()

	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	torch.manual_seed(0)

	parser = argparse.ArgumentParser(description="Train a Bayesian network for brain damage recovery example.")

	parser.add_argument('--mode', type=str, choices=['h2h', 'h2c', 'c2c'], required=True, help="Mode of snapshot generation.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for results.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Output directory for checkpoints.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

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
	
	# Load snapshots

	path_train = os.path.join('snapshots', 'snapshots_train_' + args.mode + '.npz')
	if not os.path.exists(path_train):
		print(f"Training snapshots not found at {path_train}.")
		exit()
	data_train = np.load(path_train)
	N_train = data_train['mu'].shape[0]
	mu_train, u_train = data_train['mu'].astype(np.float32), data_train['u'].astype(np.float32)
	mu_train, u_train = torch.tensor(mu_train).to(device), torch.tensor(u_train).to(device)

	path_test = os.path.join('snapshots', 'snapshots_test_' + args.mode + '.npz')
	if not os.path.exists(path_test):
		print(f"Test snapshots not found at {path_test}.")
		exit()
	data_test = np.load(path_test)
	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Deterministic network definition

	layer_1 = Geodesic(domain, Vh_H, Vh_C, support=0.05) # default activation: leakyReLU
	layer_2 = Geodesic(domain, Vh_C, Vh_C, support=0.1)
	layer_3 = Geodesic(domain, Vh_C, Vh_H, support=0.05, activation=None)

	model = DFNN(layer_1, layer_2, layer_3)
	model.He()

	if device.type == 'cuda':
		model.cuda()
	else:
		model.cpu()

	# Deterministic network training

	model.train(mu_train, u_train, ntrain=N_train, epochs=100, loss=mse(euclidean), verbose=True)

	# Deterministic network evaluation

	model.eval()

	with torch.no_grad():
		u_pred_train = model(mu_train)
		u_pred = model(mu_test)

	error_train = torch.norm(u_train - u_pred_train, dim=1) / torch.norm(u_train, dim=1)
	error_test = torch.norm(u_test - u_pred, dim=1) / torch.norm(u_test, dim=1)
	print('Relative training error: {:.2f}%'.format(100 * torch.mean(error_train)))
	print('Relative test error: {:.2f}%'.format(100 * torch.mean(error_test)))

	# Save figures

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	u_pred = u_pred.detach().cpu().numpy()

	for idx in range(N_test):
		plt.figure(figsize=(12, 5))
		plt.subplot(1, 3, 1)
		plt.title("Brain damage")
		fe.plot(1 + 0 * mu_test[idx], Vh_H, cmap='jet', vmin=0, vmax=1)
		fe.plot(mu_test[idx], Vh_H, cmap='jet')
		plt.subplot(1, 3, 2)
		plt.title("True time to recovery")
		fe.plot(u_test[idx], Vh_H, cmap='jet')
		plt.subplot(1, 3, 3)
		plt.title("Predicted time to recovery")
		fe.plot(u_pred[idx], Vh_H, cmap='jet')

	# Save model
	torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model_deterministic.pth'))