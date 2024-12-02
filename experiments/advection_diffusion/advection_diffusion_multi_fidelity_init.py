import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *
from dlroms.minns import Interpolate
from dlroms_bayesian.expansions import ExpandedLocal
from dlroms_bayesian.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

	# Domain and mesh definition

	mesh_H = fe.unitsquaremesh(100, 100) # fine mesh
	V_H = fe.space(mesh_H, 'CG', 1) 
	Nh_H = V_H.dim()

	mesh_C = fe.unitsquaremesh(50, 50) # coarse mesh
	V_C = fe.space(mesh_C, 'CG', 1)
	Nh_C = V_C.dim()

	# Load snapshots

	path_train_H = os.path.join(args.snapshot_dir, 'snapshots_train_H_' + str(args.num_snapshots_high) + '.npz')
	data_train_H = np.load(path_train_H)

	mu_train_H, u_train_H = data_train_H['mu'].astype(np.float32), data_train_H['u'].astype(np.float32)
	mu_train_H, u_train_H = torch.tensor(mu_train_H).to(device), torch.tensor(u_train_H).to(device)

	path_train_C = os.path.join(args.snapshot_dir, 'snapshots_train_C_' + str(args.num_snapshots_low) + '.npz')
	data_train_C = np.load(path_train_C)

	mu_train_C, u_train_C = data_train_C['mu'].astype(np.float32), data_train_C['u'].astype(np.float32)
	mu_train_C, u_train_C = torch.tensor(mu_train_C).to(device), torch.tensor(u_train_C).to(device)

	path_test = os.path.join(args.snapshot_dir, 'snapshots_test.npz')
	data_test = np.load(path_test)

	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Traning architecture

	m = 16
	k = 4

	psi_prime = Dense(Nh_C, 4, activation=None)

	psi = Dense(4, 100 * m) + \
			Reshape(4 * m, 5, 5) + \
			Deconv2D(7, (4 * m, 2 * m), 1) + \
			Deconv2D(4, (2 * m, m), 2) + \
			Deconv2D(5, (m, 1), 2, activation=None) + \
			Reshape(-1)

	phi = Dense(4, 50 * k) + \
			Dense(50 * k, 50 * k) + \
			Dense(50 * k, 4, activation=None)

	print("Trainable parameters:")
	print("\tEncoder:", psi_prime.dof())
	print("\tDecoder:", psi.dof())
	print("\tDense NN:", phi.dof())
	print("\tMesh-informed layer:", chi.dof())

	# Load the dense NN and the decoder

	psi.load(os.path.join('checkpoints', 'psi_' + str(args.num_snapshots_low) + '_' + str(args.num_snapshots_high)))
	phi.load(os.path.join('checkpoints', 'phi_' + str(args.num_snapshots_low) + '_' + str(args.num_snapshots_high)))

	psi.freeze()
	phi.freeze()

	if args.init == 'he':
		chi = Local(V_C, V_H, support=0.1, activation=None)
		chi.He()
	elif args.init == 'det':
		chi = ExpandedLocal(V_C, V_H, support=0.1, activation=None)
		chi.deterministic()
	elif args.init == 'interp':
		chi = Interpolate(mesh_C, V_C, V_H)

	model = DFNN(phi, psi, chi)

	if torch.cuda.is_available():
		model.cuda()

	model.train(mu_train_H, u_train_H, ntrain=args.num_snapshots_high, epochs=40, loss=mse(euclidean), verbose=True)

	model.eval()

	with torch.no_grad():
		u_train_H_pred = model(mu_train_H)

	error_train = mre(euclidean)(u_train_H, u_train_H_pred)
	print(f"Relative training error: {100 * torch.mean(error_train):.2f}%")

	with torch.no_grad():
		u_test_pred = model(mu_test)

	error_test = mre(euclidean)(u_test, u_test_pred)
	print(f"Relative test error: {100 * torch.mean(error_test):.2f}%")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train multi-fidelity model for advection-diffusion example (mesh-informed layer only).")

	parser.add_argument('--init', type=str, choices=['he', 'det', 'interp'], default='he', help="Initialization strategy.")
	parser.add_argument('--num_snapshots_high', type=int, default=75, help="Number of high-fidelity training snapshots.")
	parser.add_argument('--num_snapshots_low', type=int, default=1200, help="Number of low-fidelity training snapshots.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")

	args = parser.parse_args()

	train(args)
