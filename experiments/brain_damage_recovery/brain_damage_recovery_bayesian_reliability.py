import os
import time
import argparse
import numpy as np
import torch
import gmsh
import sys

from dlroms import *
from dlroms_bayesian.bayesian import Bayesian
from dlroms_bayesian.svgd import SVGD


if __name__ == '__main__':

	gmsh.initialize()

	parser = argparse.ArgumentParser(description="Plot reliability diagram for the Bayesian network.")

	parser.add_argument('--init', type=str, choices=['he', 'hyb'], required=True, help="Initialization strategy (He or hybrid).")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results_bayesian', help="Output directory for results.")

	args = parser.parse_args()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

	ns_samples = [10, 30, 50, 70]
	alphas = np.linspace(0.01, 0.99, 50)
	
	u_mean_coverages = []

	for n_samples in ns_samples:

		print(f"Computing reliability diagram of mean time-to-recovery with {n_samples} samples...")

		u_mean_coverage = []

		trainer = SVGD(model_bayes, n_samples=n_samples)
		model_bayes.set_trainer(trainer)

		if args.init == 'he':
			trainer.load_particles(os.path.join(args.checkpoint_dir, "particles_" + str(n_samples) + ".pth"))
		elif args.init == 'hyb':
			trainer.load_particles(os.path.join(args.checkpoint_dir, "particles_" + str(n_samples) + "_hyb.pth"))

		with torch.no_grad():
			u_pred_full = model_bayes(mu_test, reduce=False) # (N_samples, N_test, nh_H)

		for alpha in alphas:
			u_mean = torch.mean(u_test, dim=1)
			u_mean_full = torch.mean(u_pred_full, dim=2) # (N_samples, N_test)
			u_mean_pred_lower = torch.quantile(u_mean_full, alpha / 2, dim=0)
			u_mean_pred_upper = torch.quantile(u_mean_full, 1 - alpha / 2, dim=0)
			coverage = torch.mean(torch.logical_and(u_mean >= u_mean_pred_lower, u_mean <= u_mean_pred_upper), dtype=float).item()
			u_mean_coverage.append(coverage)

		model_bayes.trainer = None
		del trainer

		u_mean_coverages.append(u_mean_coverage)

	plt.figure(figsize=(12, 9))
	plt.title(f"Reliablity diagram, mean time-to-recovery", fontsize=20)
	for i, n_samples in enumerate(ns_samples):
		plt.plot(1 - alphas, u_mean_coverages[i], label=f"{n_samples} samples")
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel("Reference", fontsize=16)
	plt.xticks(fontsize=16)
	plt.ylabel("Observed", fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=16)
	plt.grid()
	plt.tight_layout()
	if args.init == 'he':
		plt.savefig(os.path.join(args.output_dir, "reliability_mean_time_to_recovery.png"))
	elif args.init == 'hyb':
		plt.savefig(os.path.join(args.output_dir, "reliability_mean_time_to_recovery_hyb.png"))

	u_max_coverages = []

	for n_samples in ns_samples:

		print(f"Computing reliability diagram of maximum time-to-recovery with {n_samples} samples...")

		u_max_coverage = []

		trainer = SVGD(model_bayes, n_samples=n_samples)
		model_bayes.set_trainer(trainer)

		if args.init == 'he':
			trainer.load_particles(os.path.join(args.checkpoint_dir, "particles_" + str(n_samples) + ".pth"))
		elif args.init == 'hyb':
			trainer.load_particles(os.path.join(args.checkpoint_dir, "particles_" + str(n_samples) + "_hyb.pth"))

		with torch.no_grad():
			u_pred_full = model_bayes(mu_test, reduce=False) # (N_samples, N_test, nh_H)

		for alpha in alphas:
			u_max = torch.max(u_test, dim=1).values
			u_max_full = torch.max(u_pred_full, dim=2).values # (N_samples, N_test)
			u_max_pred_lower = torch.quantile(u_max_full, alpha / 2, dim=0)
			u_max_pred_upper = torch.quantile(u_max_full, 1 - alpha / 2, dim=0)
			coverage = torch.mean(torch.logical_and(u_max >= u_max_pred_lower, u_max <= u_max_pred_upper), dtype=float).item()
			u_max_coverage.append(coverage)

		model_bayes.trainer = None
		del trainer

		u_max_coverages.append(u_max_coverage)

	plt.figure(figsize=(12, 9))
	plt.title(f"Reliablity diagram, maximum time-to-recovery", fontsize=20)
	for i, n_samples in enumerate(ns_samples):
		plt.plot(1 - alphas, u_max_coverages[i], label=f"{n_samples} samples")
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel("Reference", fontsize=16)
	plt.xticks(fontsize=16)
	plt.ylabel("Observed", fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=16)
	plt.grid()
	plt.tight_layout()
	if args.init == 'he':
		plt.savefig(os.path.join(args.output_dir, "reliability_max_time_to_recovery.png"))
	elif args.init == 'hyb':
		plt.savefig(os.path.join(args.output_dir, "reliability_max_time_to_recovery_hyb.png"))
