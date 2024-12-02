import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from dolfin import *
from dlroms import *
from dlroms.dnns import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

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

    # Load train and test data

    path_train_C = os.path.join(os.getcwd(), args.snapshot_dir, "snapshots_train_C.npz")
    data_train_C = np.load(path_train_C)

    N_train_C = data_train_C['K'].shape[0]
    K_train_C = torch.tensor(data_train_C['K'].astype(np.float32)).to(device)
    out_train_C = torch.tensor(data_train_C[args.field].astype(np.float32)).to(device)

    path_test_C = os.path.join(os.getcwd(), args.snapshot_dir, "snapshots_test_C.npz")
    data_test_C = np.load(path_test_C)

    N_test_C = data_test_C['K'].shape[0]
    K_test_C = torch.tensor(data_test_C['K'].astype(np.float32)).to(device)
    out_test_C = torch.tensor(data_test_C[args.field].astype(np.float32)).to(device)

    path_train_H = os.path.join(os.getcwd(), args.snapshot_dir, "snapshots_train_H.npz")
    data_train_H = np.load(path_train_H)

    N_train_H = data_train_H['K'].shape[0]
    K_train_H = torch.tensor(data_train_H['K'].astype(np.float32)).to(device)
    out_train_H = torch.tensor(data_train_H[args.field].astype(np.float32)).to(device)

    path_test_H = os.path.join(os.getcwd(), args.snapshot_dir, "snapshots_test_H.npz")
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

    # Train encoder-decoder network

    model = DFNN(psi, psi_prime)
    model.He()

    if torch.cuda.is_available():
        model.cuda()

    model.train(K_train_C, out_train_C, ntrain=N_train_C, epochs=200, loss=mse(l2_C), verbose=True)

    # From low fidelity to high fidelity data with mesh-informed layer

    model.freeze() # freeze the weights of the trained encoder-decoder network

    layer_in.He() # reset weights
    layer_out.He()

    model_refined = DFNN(layer_in, model, layer_out)

    if torch.cuda.is_available():
        model_refined.cuda()

    model_refined.train(K_train_H, out_train_H, ntrain=N_train_H, epochs=100, loss=mse(l2_H), verbose=True)

    # Generate predictions

    with torch.no_grad():
        pred_train = model(K_train_C)
        pred = model(K_test_C)
        pred_train_refined = model_refined(K_train_H)
        pred_refined = model_refined(K_test_H)

    # Compute relative error

    error_train = mre(l2_C)(out_train_C, pred_train)
    error_test = mre(l2_C)(out_test_C, pred)
    error_train_refined = mre(l2_H)(out_train_H, pred_train_refined)
    error_test_refined = mre(l2_H)(out_test_H, pred_refined)

    print(f"Relative training error, low-res.: {100 * torch.mean(error_train):.2f}%")
    print(f"Relative test error, low-res.: {100 * torch.mean(error_test):.2f}%")
    print(f"Relative training error, high-res.: {100 * torch.mean(error_train_refined):.2f}%")
    print(f"Relative test error, high-res.: {100 * torch.mean(error_test_refined):.2f}%")

    # Save trained model

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.field + '_model_C.pth'))
    torch.save(model_refined.state_dict(), os.path.join(args.checkpoint_dir, args.field + '_model_H.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train models for Darcy flow example.")

    parser.add_argument('--field', type=str, choices=['p', 'u_x', 'u_y'], required=True, help="Output field.")
    parser.add_argument('--snapshot_dir', type=str, default='snapshots', help="Directory containing snapshots.")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory containing model checkpoints.")

    args = parser.parse_args()

    train(args)
