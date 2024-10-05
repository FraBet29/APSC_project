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

	# parser.add_argument('--mode', type=str, choices=['h2h', 'h2c', 'c2c'], required=True, help="Mode of snapshot generation.")
	parser.add_argument('--field', choices=['p', 'u'], required=True, help="Output field.")
	parser.add_argument('--output_dir', type=str, default='snapshots', help="Output directory for results.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Output directory for checkpoints.")
	parser.add_argument('--verbose', action='store_true', help="Verbose output.")

	args = parser.parse_args()

	# Domain and mesh definition

	domain = fe.rectangle((0.0, 0.0), (1.0, 1.0))
	mesh = fe.mesh(domain, stepsize=0.05)

	# Load snapshots

	path_train = os.path('snapshots')
	if not os.path.exists(path_train):
		print(f"Training snapshots not found at {path_train}.")
		exit()
	
	K_train = np.load(os.path.join(path_train, 'K_train.npy'))
	p_train = np.load(os.path.join(path_train, 'p_train.npy'))
	u_x_train = np.load(os.path.join(path_train, 'u_x_train.npy'))
	u_y_train = np.load(os.path.join(path_train, 'u_y_train.npy'))
	N_train = K_train.shape[0]

	K_test = np.load(os.path.join(path_train, 'K_test.npy'))
	p_test = np.load(os.path.join(path_train, 'p_test.npy'))
	u_x_test = np.load(os.path.join(path_train, 'u_x_test.npy'))
	u_y_test = np.load(os.path.join(path_train, 'u_y_test.npy'))
	N_test = K_test.shape[0]

	# TODO: Implement Bayesian network training (either for p or u)