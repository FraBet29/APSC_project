import numpy as np
import torch
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(args):

	# Domain, mesh, and function space definition

	domain = fe.rectangle((0.0, 0.0), (1.0, 1.0))

	mesh_C = fe.mesh(domain, stepsize=0.05)
	mesh_H = fe.mesh(domain, stepsize=0.02)

	V_C = fe.space(mesh_C, 'CG', 1) # 441 dofs
	V_H = fe.space(mesh_H, 'CG', 1) # 2601 dofs

	l2_C = L2(V_C) # L2 norm
	l2_H = L2(V_H)

	if torch.cuda.is_available():
		l2_C.cuda()
		l2_H.cuda()

	# Load test data

	path_test_H = os.path.join(args.snapshot_dir, "snapshots_test_H.npz")
	data_test_H = np.load(path_test_H)

	N_test_H = data_test_H['K'].shape[0]
	K_test_H = torch.tensor(data_test_H['K'].astype(np.float32)).to(device)
	out_test_H = torch.tensor(data_test_H[args.field].astype(np.float32)).to(device)

	# Neural network cores

	m = 16

	# Encoder
	psi = Reshape(1, 21, 21) + \
		Conv2D(6, (1, m), stride=1, activation=torch.tanh) + \
		Conv2D(7, (m, 2 * m), stride=1, activation=torch.tanh) + \
		Conv2D(7, (2 * m, 4 * m), stride=1, activation=None)

	# Decoder
	psi_prime = Deconv2D(7, (4 * m, 2 * m), stride=1, activation=torch.tanh) + \
				Deconv2D(7, (2 * m, m), stride=1, activation=torch.tanh) + \
				Deconv2D(6, (m, 1), stride=1, activation=None) + \
				Reshape(-1)

	# Mesh-informed layers
	layer_in = Local(V_H, V_C, support=0.05, activation=None)
	layer_out = Local(V_C, V_H, support=0.05, activation=None)

	# Model evaluation

	model = DFNN(psi, psi_prime)
	model_refined = DFNN(layer_in, model, layer_out)
	model_refined.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.field + "_model_H.pth"), map_location=device, weights_only=True))

	if torch.cuda.is_available():
		model_refined.cuda()

	with torch.no_grad():
		pred_refined = model_refined(K_test_H)

	error_refined = mre(l2_H)(out_test_H, pred_refined)

	print(f"Test error (high-res.): {100 * error_refined:.2f}%")

	# Save results

	if args.save_all:

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		
		for idx in range(N_test_H):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(12, 4))
			plt.subplot(1, 3, 1)
			fe.plot(K_test_H[idx], V_H, cmap='jet', colorbar=True)
			plt.title("K")
			vmin, vmax = torch.min(out_test_H[idx]), torch.max(out_test_H[idx])
			plt.subplot(1, 3, 2)
			fe.plot(out_test_H[idx], V_H, cmap='jet', vmin=vmin, vmax=vmax, colorbar=True)
			plt.title(f"True {args.field}")
			plt.subplot(1, 3, 3)
			fe.plot(pred_refined[idx], V_H, cmap='jet', vmin=vmin, vmax=vmax, colorbar=True)
			plt.title(f"Predicted {args.field}")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"{args.field}_{idx}.png"))

			plt.close('all')

		print(f"Results saved in directory {args.output_dir}.")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Evaluate trained models for Darcy flow example.")

	parser.add_argument('--field', type=str, choices=['p', 'u_x', 'u_y'], required=True, help="Output field.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	evaluate(args)
