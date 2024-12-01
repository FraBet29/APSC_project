import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

from dlroms_bayesian.bayesian import Bayesian
from dlroms_bayesian.svgd import SVGD
from dlroms_bayesian.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

    # Domain and mesh definition

    domain = fe.rectangle((0.0, 0.0), (1.0, 1.0))
    mesh = fe.mesh(domain, stepsize=0.05)
    V = fe.space(mesh, 'CG', 1) # 441 dofs
    l2 = L2(V) # L2 norm

    if torch.cuda.is_available():
        l2.cuda()

    # Load train and test data

    path_train = os.path.join(os.getcwd(), "snapshots", "snapshots_train_C.npz")
    data_train = np.load(path_train)

    N_train = data_train['K'].shape[0]
    K_train = torch.tensor(data_train['K'].astype(np.float32)).to(device)
    out_train = torch.tensor(data_train[args.field].astype(np.float32)).to(device)

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

    print("Encoder trainable parameters:", psi.dof())
    print("Decoder trainable parameters:", psi_prime.dof())

    # Train Bayesian network

    model = DFNN(psi, psi_prime)

    bayes = Bayesian(model)

    if torch.cuda.is_available():
        bayes.cuda()

    N_particles = 10

    p_trainer = SVGD(bayes, n_samples=N_particles)
    p_trainer.He()
    bayes.set_trainer(p_trainer)

    # bayes.train(K_train, out_train, ntrain=N_train, lr=1e-3, lr_noise=1e-3, loss=mse(l2), epochs=5000)
    history = bayes.train(K_train, out_train, ntrain=N_train, lr=1e-3, lr_noise=1e-3, loss=mse(l2), epochs=5000, track_history=True)

    # Compute mean and variance of predictions

    pred_bayes_mean_train, pred_bayes_var_train = bayes.sample(K_train, n_samples=N_particles)
    pred_bayes_mean, pred_bayes_var = bayes.sample(K_test, n_samples=N_particles)

    # Compute relative error

    error_train = mre(l2)(out_train, pred_bayes_mean_train)
    error_test = mre(l2)(out_test, pred_bayes_mean)

    print(f"Relative training error: {100 * torch.mean(error_train):.2f}")
    print(f"Relative test error: {100 * torch.mean(error_test):.2f}")

    # Save trainer state

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    p_trainer.save_particles(os.path.join(args.checkpoint_dir, args.field + '_particles.pth'))

    # Save training history

    import pickle

    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Bayesian models for Darcy flow example.")

    parser.add_argument('--field', type=str, choices=['p', 'u_x', 'u_y'], required=True, help="Output field.")
    parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")

    args = parser.parse_args()

    set_seeds(0) # set random seed for reproducibility

    train(args)
