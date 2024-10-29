import os
import time
import argparse
import numpy as np
from dlroms import *
import torch
import gmsh
import sys
sys.path.append(os.path.join("..", "..", "dlroms")) # TODO: better alternative?
from bayesian import *


if __name__ == '__main__':

	gmsh.initialize()

	parser = argparse.ArgumentParser(description="Evaluate trained models for Darcy flow example.")

	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results_bayesian', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
	if not os.path.exists(path_test):
		print(f"Test snapshots not found at {path_test}.")
		exit()
	data_test = np.load(path_test)
	N_test = data_test['mu'].shape[0]
	mu_test, u_test = data_test['mu'].astype(np.float32), data_test['u'].astype(np.float32)
	mu_test, u_test = torch.tensor(mu_test).to(device), torch.tensor(u_test).to(device)

	# Uncertainty quantification

	N_samples = 70 # number of samples from the posterior
	alpha = 0.1 # confidence level

	# Bayesian network definition

	layer_1 = Geodesic(domain, Vh_H, Vh_C, support=0.05) # default activation: leakyReLU
	layer_2 = Geodesic(domain, Vh_C, Vh_C, support=0.1)
	layer_3 = Geodesic(domain, Vh_C, Vh_H, support=0.05, activation=None)

	model = DFNN(layer_1, layer_2, layer_3)

	if torch.cuda.is_available():
		model.cuda()

	model_bayes = Bayesian(model)

	if torch.cuda.is_available():
		model_bayes.cuda()

	trainer = SVGD(model_bayes, n_samples=N_samples)
	model_bayes.set_trainer(trainer)
	trainer.load_particles(os.path.join(args.checkpoint_dir, "particles_" + str(N_samples) + ".pth"))

	with torch.no_grad():
		u_pred_bayes_mean, u_pred_bayes_var = model_bayes.sample(mu_test, n_samples=N_samples)

	# Compute relative test error

	error_test_mean = mre(l2)(u_test, u_pred_bayes_mean)
	print(f"Relative test error: {100 * torch.mean(error_test_mean):.2f}%")

	# Compute mean time-to-recovery

	with torch.no_grad():
		u_pred_full = model_bayes(mu_test) # (N_samples, N_test, nh_H)

	u_mean = torch.mean(u_test, dim=1)
	u_mean_full = torch.mean(u_pred_full, dim=2) # (N_samples, N_test)

	# Confidence interval for mean time-to-recovery
	u_mean_pred_lower = torch.quantile(u_mean_full, alpha / 2, dim=0)
	u_mean_pred_upper = torch.quantile(u_mean_full, 1 - alpha / 2, dim=0)
	u_mean_pred_median = torch.median(u_mean_full, dim=0).values

	# Coverage
	coverage = torch.mean(torch.logical_and(u_mean >= u_mean_pred_lower, u_mean <= u_mean_pred_upper), dtype=float).item()
	print(f"Mean time to recovery coverage ({N_samples} samples, {1 - alpha:.0%} confidence): {100 * coverage:.2f}%")

	plt.figure(figsize=(12, 5))
	plt.title(f"Mean time to recovery ({N_samples} samples)")
	plt.plot(u_mean, 'b', label='True')
	plt.plot(u_mean_pred_median, 'r')
	plt.fill_between(range(N_test), u_mean_pred_lower.cpu(), u_mean_pred_upper.cpu(), color='r', alpha=0.2)
	plt.xlabel("Snapshot index")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(args.output_dir, "mean_recovery_time_" + str(N_samples) + ".png"))

	# Compute maximum time-to-recovery

	with torch.no_grad():
		u_pred_full = model_bayes(mu_test) # (N_samples, N_test, nh_H)

	u_max = torch.max(u_test, dim=1).values
	u_max_full = torch.max(u_pred_full, dim=2).values # (N_samples, N_test)

	# Confidence interval for maximum time-to-recovery
	u_max_pred_lower = torch.quantile(u_max_full, alpha / 2, dim=0)
	u_max_pred_upper = torch.quantile(u_max_full, 1 - alpha / 2, dim=0)
	u_max_pred_median = torch.median(u_max_full, dim=0).values

	# Coverage
	coverage = torch.mean(torch.logical_and(u_max >= u_max_pred_lower, u_max <= u_max_pred_upper), dtype=float).item()
	print(f"Maximum time to recovery coverage ({N_samples} samples, {1 - alpha:.0%} confidence): {100 * coverage:.2f}%")

	plt.figure(figsize=(12, 5))
	plt.title(f"Maximum time to recovery ({N_samples} samples)")
	plt.plot(u_max, 'b', label='True')
	plt.plot(u_max_pred_median, 'r')
	plt.fill_between(range(N_test), u_max_pred_lower.cpu(), u_max_pred_upper.cpu(), color='r', alpha=0.2)
	plt.xlabel("Snapshot index")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(args.output_dir, "max_recovery_time_" + str(N_samples) + ".png"))

	if args.save_all:
		
		u_pred_bayes_mean, u_pred_bayes_var = u_pred_bayes_mean.detach().cpu().numpy(), u_pred_bayes_var.detach().cpu().numpy()

		for idx in range(N_test):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(16, 5))
			plt.subplot(1, 4, 1)
			plt.title("Brain damage")
			fe.plot(1 + 0 * mu_test[idx], Vh_H, cmap='jet', vmin=0, vmax=1)
			fe.plot(mu_test[idx], Vh_H, cmap='jet')
			plt.subplot(1, 4, 2)
			plt.title("True time to recovery")
			fe.plot(u_test[idx], Vh_H, cmap='jet')
			plt.subplot(1, 4, 3)
			plt.title("Predicted time to recovery (mean)")
			fe.plot(u_pred_bayes_mean[idx], Vh_H, cmap='jet')
			plt.subplot(1, 4, 4)
			plt.title("Predicted time to recovery (variance)")
			fe.plot(u_pred_bayes_var[idx], Vh_H, cmap='jet')
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"result_{idx}.png"))
