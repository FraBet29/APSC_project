import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dlroms import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(args):

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

	l2 = L2(Vh_H)

	if torch.cuda.is_available():
		l2.cuda()

	# Load test snapshots

	path_test = os.path.join(args.snapshot_dir, 'snapshots_test.npz')
	data_test = np.load(path_test)

	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Mhes-informed network definition

	layer_1 = Geodesic(domain, Vh_H, Vh_C, support=0.05) # default activation: leakyReLU
	layer_2 = Geodesic(domain, Vh_C, Vh_C, support=0.1)
	layer_3 = Geodesic(domain, Vh_C, Vh_H, support=0.05, activation=None)

	model = DFNN(layer_1, layer_2, layer_3)
	model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model' + filetag + '.pth'), weights_only=True, map_location=device))

	if torch.cuda.is_available():
		model.cuda()

	with torch.no_grad():
		u_pred = model(mu_test)

	# Compute relative test error

	error_test = mre(l2)(u_test, u_pred)
	print(f"Relative test error: {100 * error_test:.2f}%")

	# Compute mean time-to-recovery

	u_mean = torch.mean(u_test, dim=1)
	u_mean_pred = torch.mean(u_pred, dim=1)

	# Compute maximum time-to-recovery

	u_max = torch.max(u_test, dim=1).values
	u_max_pred = torch.max(u_pred, dim=1).values

	if args.save_all:

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)

		plt.figure(figsize=(12, 5))
		plt.title("Mean time-to-recovery")
		plt.plot(u_mean.cpu().numpy(), 'b', label='True')
		plt.plot(u_mean_pred.cpu().numpy(), 'r', label='Predicted')
		plt.xlabel("Snapshot index")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(args.output_dir, 'mean_recovery_time' + filetag + '.png'))
		plt.close()

		plt.figure(figsize=(12, 5))
		plt.title("Maximum time-to-recovery")
		plt.plot(u_max.cpu().numpy(), 'b', label='True')
		plt.plot(u_max_pred.cpu().numpy(), 'r', label='Predicted')
		plt.xlabel("Snapshot index")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(args.output_dir, 'max_recovery_time' + filetag + '.png'))
		plt.close()
		
		for idx in range(N_test):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(16, 4))
			plt.subplot(1, 3, 1)
			plt.title("Brain damage")
			fe.plot(1 + 0 * mu_test[idx], Vh_H, cmap='jet', vmin=0, vmax=1)
			fe.plot(mu_test[idx], Vh_H, cmap='jet', colorbar=True)
			plt.subplot(1, 3, 2)
			plt.title("True time-to-recovery")
			fe.plot(u_test[idx], Vh_H, cmap='jet', vmin=0, vmax=1, colorbar=True)
			plt.subplot(1, 3, 3)
			plt.title("Predicted time-to-recovery")
			fe.plot(u_pred[idx], Vh_H, cmap='jet', vmin=0, vmax=1, colorbar=True)
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f'brain_{idx}.png'))
			plt.close()

			print(f"Results saved in directory {args.output_dir}.")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Evaluate trained models for brain damage recovery example.")

	parser.add_argument('--init', type=str, choices=['he', 'det', 'hyb'], required=True, help="Initialization strategy.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	evaluate(args)
