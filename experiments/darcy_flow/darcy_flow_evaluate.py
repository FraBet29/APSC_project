from dlroms import *
from dlroms.dnns import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Evaluate trained models for Darcy flow example.")

	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
	p_test_H = torch.tensor(data_test_H['p'].astype(np.float32)).to(device)
	u_x_test_H = torch.tensor(data_test_H['u_x'].astype(np.float32)).to(device)
	u_y_test_H = torch.tensor(data_test_H['u_y'].astype(np.float32)).to(device)

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

	# Pressure model

	p_model = DFNN(psi, psi_prime)
	p_model_refined = DFNN(layer_in, p_model, layer_out)
	p_model_refined.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "p_model_H.pth"), map_location=device))

	if torch.cuda.is_available():
		p_model_refined.cuda()

	with torch.no_grad():
		p_pred_refined = p_model_refined(K_test_H)

	# Velocity model

	u_x_model = DFNN(psi, psi_prime)
	u_x_model_refined = DFNN(layer_in, u_x_model, layer_out)
	u_x_model_refined.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "u_x_model_H.pth"), map_location=device))

	if torch.cuda.is_available():
		u_x_model_refined.cuda()
	
	with torch.no_grad():
		u_x_pred_refined = u_x_model_refined(K_test_H)

	u_y_model = DFNN(psi, psi_prime)
	u_y_model_refined = DFNN(layer_in, u_y_model, layer_out)
	u_y_model_refined.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "u_y_model_H.pth"), map_location=device))

	if torch.cuda.is_available():
		u_y_model_refined.cuda()

	with torch.no_grad():
		u_y_pred_refined = u_y_model_refined(K_test_H)

	u_test_H = torch.sqrt(u_x_test_H ** 2 + u_y_test_H ** 2)
	u_pred_refined = torch.sqrt(u_x_pred_refined ** 2 + u_y_pred_refined ** 2)

	# Compute errors

	p_error_refined = mre(l2_H)(p_test_H, p_pred_refined)
	u_x_error_refined = mre(l2_H)(u_x_test_H, u_x_pred_refined)
	u_y_error_refined = mre(l2_H)(u_y_test_H, u_y_pred_refined)
	u_error_refined = mre(l2_H)(u_test_H, u_pred_refined)

	print(f"Pressure error (refined model): {100 * p_error_refined:.2f}%")
	print(f"Velocity error, x-component (refined model): {100 * u_x_error_refined:.2f}%")
	print(f"Velocity error, y-component (refined model): {100 * u_y_error_refined:.2f}%")
	print(f"Velocity error, magnitude (refined model): {100 * u_error_refined:.2f}%")

	# Save results

	if args.save_all:

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		
		for idx in range(N_test_H):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(p_test_H[idx], V_H, cmap='jet', colorbar=True)
			plt.title("True pressure")
			plt.subplot(1, 2, 2)
			fe.plot(p_pred_refined[idx], V_H, cmap='jet', colorbar=True)
			plt.title("Predicted pressure")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"p_{idx}.png"))

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(u_x_test_H[idx], V_H, cmap='jet', colorbar=True)
			plt.title("True velocity (x-comp.)")
			plt.subplot(1, 2, 2)
			fe.plot(u_x_pred_refined[idx], V_H, cmap='jet', colorbar=True)
			plt.title("Predicted velocity (x-comp.)")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"u_x_{idx}.png"))

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(u_y_test_H[idx], V_H, cmap='jet', colorbar=True)
			plt.title("True velocity (y-comp.)")
			plt.subplot(1, 2, 2)
			fe.plot(u_y_pred_refined[idx], V_H, cmap='jet', colorbar=True)
			plt.title("Predicted velocity (y-comp.)")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"u_y_{idx}.png"))

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(u_test_H[idx], V_H, cmap='jet', colorbar=True)
			plt.title("True velocity (magnitude)")
			plt.subplot(1, 2, 2)
			fe.plot(u_pred_refined[idx], V_H, cmap='jet', colorbar=True)
			plt.title("Predicted velocity (magnitude)")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"u_{idx}.png"))

			plt.close('all')

		print(f"Results saved in directory {args.output_dir}.")
