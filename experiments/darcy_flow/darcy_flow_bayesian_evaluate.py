import numpy as np
import torch
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

from dlroms_bayesian.bayesian import Bayesian
from dlroms_bayesian.svgd import SVGD
from dlroms_bayesian.utils import *

import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(args):

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
	out_test = torch.tensor(data_test[args.field].astype(np.float32)).to(device)	

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

	# Evaluate Bayesian model

	model = DFNN(psi, psi_prime)

	bayes = Bayesian(model)

	if torch.cuda.is_available():
		bayes.cuda()

	N_particles = 10

	trainer = SVGD(bayes, n_samples=N_particles)
	bayes.set_trainer(trainer)
	trainer.load_particles(os.path.join(args.checkpoint_dir, args.field + "_particles.pth"))

	pred_bayes_mean, pred_bayes_var = bayes.sample(K_test, n_samples=N_particles)

	error_bayes_mean = mre(l2)(out_test, pred_bayes_mean)

	print(f"Test error (high-res.): {100 * error_bayes_mean:.2f}%")

	# Save results

	if args.save_all:

		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		
		for idx in range(N_test):

			print(f"Saving results for test sample {idx}...")

			plt.figure(figsize=(12, 3))
			plt.subplot(1, 4, 1)
			fe.plot(K_test[idx], V, cmap='jet', colorbar=True)
			plt.title("K")
			vmin, vmax = torch.min(out_test[idx]), torch.max(out_test[idx])
			plt.subplot(1, 4, 2)
			fe.plot(out_test[idx], V, cmap='jet', vmin=vmin, vmax=vmax, colorbar=True)
			plt.title(f"True {args.field}")
			plt.subplot(1, 4, 3)
			fe.plot(pred_bayes_mean[idx], V, cmap='jet', vmin=vmin, vmax=vmax, colorbar=True)
			plt.title(f"Predicted {args.field}")
			plt.subplot(1, 4, 4)
			fe.plot(pred_bayes_var[idx], V, cmap='magma', colorbar=True)
			plt.title(f"{args.field} variance")
			plt.tight_layout()
			plt.savefig(os.path.join(args.output_dir, f"{args.field}_{idx}.png"))

			plt.close('all')

		if os.path.exists(args.field + "_history.pkl"):

			import pickle 
			
			with open(args.field + "_history.pkl", "rb") as f:
				history = pickle.load(f)

			plt.figure(figsize=(8, 6))
			plt.plot(history['err'][100:]) # discard the first epochs for a better visualization
			plt.xlabel("epochs", fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.title("Training error", fontsize=20)
			plt.savefig(os.path.join(args.output_dir, args.field + "_err_history.png"))

			plt.figure(figsize=(8, 6))
			plt.plot(history['log_posterior'])
			plt.xlabel("epochs", fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.title("Log-posterior", fontsize=20)
			plt.yscale('symlog')
			min_val, max_val = min(history['log_posterior']), max(history['log_posterior'])
			plt.ylim(min_val * 0.8, max_val * 1.2)
			plt.savefig(os.path.join(args.output_dir, args.field + "_log_posterior_history.png"))

			plt.figure(figsize=(8, 6))
			plt.plot(history['grad_theta'])
			plt.xlabel("epochs", fontsize=16)
			plt.xticks(fontsize=16)
			plt.yticks(fontsize=16)
			plt.title("Gradient of theta (norm)", fontsize=20)
			plt.yscale('log')
			plt.savefig(os.path.join(args.output_dir, args.field + "_grad_theta_history.png"))
			
			plt.close('all')

		print(f"Results saved in directory {args.output_dir}.")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Evaluate trained Bayesian models for Darcy flow example.")

	parser.add_argument('--field', type=str, choices=['p', 'u_x', 'u_y'], required=True, help="Output field.")
	parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")
	parser.add_argument('--output_dir', type=str, default='results_bayesian', help="Output directory for results.")
	parser.add_argument('--save_all', action='store_true', help="Save all results.")

	args = parser.parse_args()

	evaluate(args)
