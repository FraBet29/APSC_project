import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

	# Domain and mesh definition

	mesh = fe.unitsquaremesh(100, 100)
	V = fe.space(mesh, 'CG', 1)
	Nh = V.dim()

	# Load snapshots

	path_train = os.path.join(args.snapshot_dir, 'snapshots_train_H_' + str(args.num_snapshots) + '.npz')
	data_train = np.load(path_train)

	N_train = data_train['mu'].shape[0]
	mu_train, u_train = data_train['mu'].astype(np.float32), data_train['u'].astype(np.float32)
	mu_train, u_train = torch.tensor(mu_train).to(device), torch.tensor(u_train).to(device)

	path_test = os.path.join(args.snapshot_dir, 'snapshots_test.npz')
	data_test = np.load(path_test)

	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Traning architecture

	m = 16
	k = 4

	psi_prime = Dense(Nh, 4, activation=None)

	psi = Dense(4, 100 * m) + \
			Reshape(4 * m, 5, 5) + \
			Deconv2D(11, (4 * m, 2 * m), 2) + \
			Deconv2D(10, (2 * m, m), 2) + \
			Deconv2D(11, (m, 1), 2, activation=None) + \
			Reshape(-1)

	phi = Dense(4, 50 * k) + \
			Dense(50 * k, 50 * k) + \
			Dense(50 * k, 4, activation=None)

	print("Trainable parameters:")
	print("  Encoder:", psi_prime.dof())
	print("  Decoder:", psi.dof())
	print("  Dense NN:", phi.dof())

	# Train the autoencoder (nonlinear dimensionality reduction)

	autoencoder = DFNN(psi_prime, psi) # encoder + decoder
	autoencoder.He()

	if torch.cuda.is_available():
		autoencoder.cuda()

	autoencoder.train(u_train, u_train, ntrain=N_train, epochs=200, loss=mre(euclidean), verbose=True)

	# Use the trained encoder to generate the reduced order version of the dataset

	psi_prime.eval()

	with torch.no_grad():
		u_train_ro = psi_prime(u_train)

	# Use the reduced dataset to train the dense NN mapping the parameters to the reduced order solution

	dense = DFNN(phi)
	dense.He()

	if torch.cuda.is_available():
		dense.cuda()

	dense.train(mu_train, u_train_ro, ntrain=N_train, epochs=200, loss=mse(euclidean), verbose=True)

	# Use the dense NN to predict the reduced order solution and the decoder to restore the full order solution

	phi.eval()
	psi.eval()

	with torch.no_grad():
		u_train_pred = psi(phi(mu_train))

	# Compute the relative error

	error_train = mre(euclidean)(u_train, u_train_pred)
	print(f"Relative training error: {100 * torch.mean(error_train):.2f}%")

	# Apply the model to the test set

	with torch.no_grad():
		u_test_pred = psi(phi(mu_test))

	error_test = torch.norm(u_test - u_test_pred, dim=1) / torch.norm(u_test, dim=1)
	print('Relative test error: {:.2f}%'.format(100 * torch.mean(error_test)))

	# Save encoder, decoder, and dense NN

	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	torch.save(psi_prime.state_dict(), os.path.join(args.checkpoint_dir, 'psi_prime_' + str(args.num_snapshots) + '.pth'))
	torch.save(psi.state_dict(), os.path.join(args.checkpoint_dir, 'psi_' + str(args.num_snapshots) + '.pth'))
	torch.save(phi.state_dict(), os.path.join(args.checkpoint_dir, 'phi_' + str(args.num_snapshots) + '.pth'))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Train high-fidelity model for advection-diffusion example.")

	parser.add_argument('--num_snapshots', type=int, default=1200, help="Number of training snapshots.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")

	args = parser.parse_args()

	train(args)
