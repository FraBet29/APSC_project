import numpy as np
import torch
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

from dlroms_bayesian.bayesian import Bayesian
from dlroms_bayesian.svgd import SVGD

import argparse
import os


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Evaluate trained Bayesian models for Darcy flow example.")

	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results_bayesian', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Domain and mesh definition

	domain = fe.rectangle((0.0, 0.0), (1.0, 1.0))
	mesh = fe.mesh(domain, stepsize=0.05)
	V = fe.space(mesh, 'CG', 1) # 441 dofs
	l2 = L2(V) # L2 norm

	if torch.cuda.is_available():
		l2.cuda()

	# Load test data

	path_test = os.path.join(os.getcwd(), "snapshots", "snapshots_test_C.npz")
	data_test = np.load(path_test)

	N_test = data_test['K'].shape[0]
	K_test = torch.tensor(data_test['K'].astype(np.float32)).to(device)
	p_test = torch.tensor(data_test['p'].astype(np.float32)).to(device)
	u_x_test = torch.tensor(data_test['u_x'].astype(np.float32)).to(device)
	u_y_test = torch.tensor(data_test['u_y'].astype(np.float32)).to(device)

	# Neural network cores

	m = 16

	# Encoder
	psi = Reshape(1, 21, 21) + \
		Conv2D(6, (1, m), stride=1) + \
		Conv2D(7, (m, 2 * m), stride=1) + \
		Conv2D(7, (2 * m, 4 * m), stride=1, activation=None)

	# Decoder
	psi_prime = Deconv2D(7, (4 * m, 2 * m), stride=1) + \
				Deconv2D(7, (2 * m, m), stride=1) + \
				Deconv2D(6, (m, 1), stride=1, activation=None) + \
				Reshape(-1)

	# Pressure model

	p_model = DFNN(psi, psi_prime)

	p_bayes = Bayesian(p_model)

	if torch.cuda.is_available():
		p_bayes.cuda()

	N_particles = 10

	p_trainer = SVGD(p_bayes, n_samples=N_particles)
	p_trainer.load_particles(os.path.join(args.checkpoint_dir, "p_particles.pth"))
	p_bayes.set_trainer(p_trainer)

	p_pred_bayes_mean, p_pred_bayes_var = p_bayes.sample(K_test, n_samples=N_particles)

	# Velocity models

	u_x_model = DFNN(psi, psi_prime)

	u_x_bayes = Bayesian(u_x_model)

	if torch.cuda.is_available():
		u_x_bayes.cuda()

	u_x_trainer = SVGD(u_x_bayes, n_samples=N_particles)
	u_x_trainer.load_particles(os.path.join(args.checkpoint_dir, "u_x_particles.pth"))
	u_x_bayes.set_trainer(u_x_trainer)

	u_x_pred_bayes_mean, u_x_pred_bayes_var = u_x_bayes.sample(K_test, n_samples=N_particles)

	u_y_model = DFNN(psi, psi_prime)

	u_y_bayes = Bayesian(u_y_model)

	if torch.cuda.is_available():
		u_y_bayes.cuda()

	u_y_trainer = SVGD(u_y_bayes, n_samples=N_particles)
	u_y_trainer.load_particles(os.path.join(args.checkpoint_dir, "u_y_particles.pth"))
	u_y_bayes.set_trainer(u_y_trainer)

	u_y_pred_bayes_mean, u_y_pred_bayes_var = u_y_bayes.sample(K_test, n_samples=N_particles)

	# Compute errors

	p_error_bayes_mean = mre(l2)(p_test, p_pred_bayes_mean)
	u_x_error_bayes_mean = mre(l2)(u_x_test, u_x_pred_bayes_mean)
	u_y_error_bayes_mean = mre(l2)(u_y_test, u_y_pred_bayes_mean)

	print(f"Pressure error (high-res.): {100 * p_error_bayes_mean:.2f}%")
	print(f"Velocity error, x-component (high-res.): {100 * u_x_error_bayes_mean:.2f}%")
	print(f"Velocity error, y-component (high-res.): {100 * u_y_error_bayes_mean:.2f}%")

	# Save results

	if args.save_all:

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		
		for idx in range(N_test):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(p_test[idx], V, cmap='jet', colorbar=True)
			plt.title("True pressure")
			plt.subplot(1, 2, 2)
			fe.plot(p_pred_bayes_mean[idx], V, cmap='jet', colorbar=True)
			plt.title("Predicted pressure")
			fe.plot(p_pred_bayes_var[idx], V, cmap='jet', colorbar=True)
			plt.title("Pressure variance")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"p_{idx}.png"))

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(u_x_test[idx], V, cmap='jet', colorbar=True)
			plt.title("True velocity (x-comp.)")
			plt.subplot(1, 2, 2)
			fe.plot(u_x_pred_bayes_mean[idx], V, cmap='jet', colorbar=True)
			plt.title("Predicted velocity (x-comp.)")
			fe.plot(u_x_pred_bayes_var[idx], V, cmap='jet', colorbar=True)
			plt.title("Velocity variance (x-comp.)")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"u_x_{idx}.png"))

			plt.figure(figsize=(12, 6))
			plt.subplot(1, 2, 1)
			fe.plot(u_y_test[idx], V, cmap='jet', colorbar=True)
			plt.title("True velocity (y-comp.)")
			plt.subplot(1, 2, 2)
			fe.plot(u_y_pred_bayes_mean[idx], V, cmap='jet', colorbar=True)
			plt.title("Predicted velocity (y-comp.)")
			fe.plot(u_y_pred_bayes_var[idx], V, cmap='jet', colorbar=True)
			plt.title("Velocity variance (y-comp.)")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"u_y_{idx}.png"))

			plt.close('all')

		if os.path.exists("history"):

			import pickle 
			
			with open(os.path.join("history", "err_history.pkl"), "rb") as f:
				err_history = pickle.load(f)
			plt.figure(figsize=(8, 6))
			plt.plot(err_history[100:]) # discard the first epochs for a better visualization
			plt.xlabel("epochs")
			plt.title("Training error")
			plt.savefig(os.path.join(args.output_dir, "err_history.png"))

			with open(os.path.join("history", "log_posterior_history.pkl"), "rb") as f:
				log_posterior_history = pickle.load(f)
			plt.figure(figsize=(8, 6))
			plt.plot(log_posterior_history)
			plt.xlabel("epochs")
			plt.title("Log-posterior")
			plt.yscale('symlog')
			plt.savefig(os.path.join(args.output_dir, "log_posterior_history.png"))

			with open(os.path.join("history", "grad_theta_history.pkl"), "rb") as f:
				grad_theta_history = pickle.load(f)
			plt.figure(figsize=(8, 6))
			plt.plot(grad_theta_history)
			plt.xlabel("epochs")
			plt.title("Gradient of theta (norm)")
			plt.yscale('log')
			plt.savefig(os.path.join(args.output_dir, "grad_theta_history.png"))
			
			plt.close('all')

		print(f"Results saved in directory {args.output_dir}.")

