import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt

from dlroms import *
from dlroms_bayesian.expansions import ExpandedGeodesic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

	filetag = '' if args.init == 'he' else '_' + args.init

	# Domain definition

	loop = lambda v: np.concatenate((v, v[[0]]))
	brain = np.load(os.path.join('brain_meshes', 'brainshape.npz'))
	domain = fe.polygon(loop(brain['main'][::9])) - fe.polygon(loop(brain['hole1'][::9])) - fe.polygon(loop(brain['hole2'][::8]))

	# Mesh and function space definition

	mesh_H = fe.loadmesh(os.path.join('brain_meshes', 'brain-mesh40.xml'))
	Vh_H = fe.space(mesh_H, 'CG', 1)
	h_H, nh_H = mesh_H.hmax(), Vh_H.dim()

	mesh_C = fe.loadmesh(os.path.join('brain_meshes', 'brain-mesh15.xml'))
	Vh_C = fe.space(mesh_C, 'CG', 1)
	h_C, nh_C = mesh_C.hmax(), Vh_C.dim()

	# Load snapshots

	path_train = os.path.join(args.snapshot_dir, 'snapshots_train.npz')
	data_train = np.load(path_train)

	N_train = data_train['mu'].shape[0]
	mu_train, u_train = data_train['mu'].astype(np.float32), data_train['u'].astype(np.float32)
	mu_train, u_train = torch.tensor(mu_train).to(device), torch.tensor(u_train).to(device)

	path_test = os.path.join(args.snapshot_dir, 'snapshots_test.npz')
	data_test = np.load(path_test)

	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Mesh-informed network training

	if args.init == 'he':
		layer_1 = Geodesic(domain, Vh_H, Vh_C, support=0.05)
		layer_2 = Geodesic(domain, Vh_C, Vh_C, support=0.1)
		layer_3 = Geodesic(domain, Vh_C, Vh_H, support=0.05, activation=None)
	elif args.init == 'det' or args.init == 'hyb':
		layer_1 = ExpandedGeodesic(domain, Vh_H, Vh_C, support=0.05)
		layer_2 = ExpandedGeodesic(domain, Vh_C, Vh_C, support=0.1)
		layer_3 = ExpandedGeodesic(domain, Vh_C, Vh_H, support=0.05, activation=None)

	l2 = L2(Vh_H)

	model = DFNN(layer_1, layer_2, layer_3)
	if args.init == 'he':
		model.He()
	elif args.init == 'det':
		for layer in model:
			layer.deterministic()
	elif args.init == 'hyb':
		for layer in model:
			layer.hybrid()

	if torch.cuda.is_available():
		model.cuda()
		l2.cuda()

	model.train(mu_train, u_train, ntrain=N_train, epochs=400, loss=mse(l2), verbose=True)

	# Mesh-informed network evaluation

	with torch.no_grad():
		u_pred_train = model(mu_train)
		u_pred = model(mu_test)

	error_train = mre(l2)(u_train, u_pred_train)
	error_test = mre(l2)(u_test, u_pred)
	print(f"Relative training error: {100 * torch.mean(error_train):.2f}%")
	print(f"Relative test error: {100 * torch.mean(error_test):.2f}%")

	# Save model

	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model' + filetag + '.pth'))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Train models for brain damage recovery example.")

	parser.add_argument('--init', type=str, choices=['he', 'det', 'hyb'], required=True, help="Initialization strategy.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")

	args = parser.parse_args()

	train(args)
