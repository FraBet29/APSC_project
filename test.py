import numpy as np
from dlroms import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import torch
from dolfin import *


# Flags

plotOn = False
generateData = False


# Domain and mesh definition

print('Generating mesh and finite element space...')

h = 0.01 													# Mesh size
n = int(1 / h)												# Number of cells

# Generate mesh with gmsh
# D = fe.rectangle((0,0), (1,1)) 							# Square domain	
# mesh = fe.mesh(D, stepsize=h) 							# Mesh (stepsize parameter available with gmsh)

# Generate mesh with FEniCS
mesh = fe.unitsquaremesh(n, n)								# Mesh

V = fe.space(mesh, 'CG', 1) 								# Continuous piecewise linear finite elements
Nh = V.dim() 												# Space dimension (number of dofs)

u_D = Expression('0.01', degree=0) 							# Function at the boundary

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary) 							# Dirichlet boundary conditions


# Sampling of the training and test sets

if generateData:

	print('Sampling training and test sets...')

	rnd.seed(42) 												# Random seed

	training_size = 1200
	test_size = 5000

	mu_train = np.zeros((training_size, 4))						# Training set

	for i in range(training_size):
		mu_train[i] = rnd.uniform(0., 1., size=(1, 4))

	mu_test = np.zeros((test_size, 4))							# Test set

	for i in range(test_size):
		mu_test[i] = rnd.uniform(0., 1., size=(1, 4))

	mu1, mu2, mu3, mu4 = mu_train[:, 0], mu_train[:, 1], mu_train[:, 2], mu_train[:, 3]
	mu1t, mu2t, mu3t, mu4t = mu_test[:, 0], mu_test[:, 1], mu_test[:, 2], mu_test[:, 3]


# Snapshots generation

if generateData:

	print('Generating snapshots...')

	u_train = np.zeros((training_size, Nh))

	for i in range(training_size):

		if i % 100 == 0:
			print('Snapshot {}/{}'.format(i, training_size))
		
		sigma = Expression('6 + 5 * tanh(20 * (x[1] + 10 * mu1 * x[0] * (x[0] - 1) * (x[0] - mu2) * (x[0] - mu3) - 0.5))', 
							degree=2, mu1=mu1[i], mu2=mu2[i], mu3=mu3[i])											# Diffusion coefficient
		
		if plotOn and i % 100 == 0: 
			plt.figure(figsize=(4, 3))
			fe.plot(interpolate(sigma, V).vector(), V, colorbar=True)												# Plot diffusion coefficient
			plt.title('sigma @ mu = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(mu1[i], mu2[i], mu3[i], mu4[i]))
			plt.show()

		beta = Expression(('10 * cos(2 * pi * mu4)', '10 * sin(2 * pi * mu4)'), degree=2, mu4=mu4[i], pi=np.pi)		# Advection coefficient

		f = Expression('100 * (x[0] * x[1] - x[1] * x[1])', degree=2)												# Source term

		u = TrialFunction(V)																						# Trial function
		v = TestFunction(V)																							# Test function
		
		F = sigma * dot(grad(u), grad(v)) * dx + dot(beta, grad(u)) * v * dx - f * v * dx							# Variational formulation
		a, L = lhs(F), rhs(F)

		u = Function(V)																								# Solution function

		solve(a == L, u, bc)																						# Solution of the variational problem

		if plotOn and i % 100 == 0:
			plt.figure(figsize=(4, 3))
			fe.plot(u.vector(), V, colorbar=True)																	# Plot solution
			plt.title('u @ mu = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(mu1[i], mu2[i], mu3[i], mu4[i]))
			plt.show()
		
		u_train[i] = u.vector()																						# Store snapshot

	u_test = np.zeros((test_size, Nh))

	for i in range(test_size):

		if i % 100 == 0:
			print('Snapshot {}/{}'.format(i, test_size))
		
		sigma = Expression('6 + 5 * tanh(20 * (x[1] + 10 * mu1 * x[0] * (x[0] - 1) * (x[0] - mu2) * (x[0] - mu3) - 0.5))', 
							degree=2, mu1=mu1t[i], mu2=mu2t[i], mu3=mu3t[i])										# Diffusion coefficient
		
		if plotOn and i % 100 == 0: 
			plt.figure(figsize=(4, 3))
			fe.plot(interpolate(sigma, V).vector(), V, colorbar=True)												# Plot diffusion coefficient
			plt.title('sigma @ mu = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(mu1t[i], mu2t[i], mu3t[i], mu4t[i]))
			plt.show()

		beta = Expression(('10 * cos(2 * pi * mu4)', '10 * sin(2 * pi * mu4)'), degree=2, mu4=mu4t[i], pi=np.pi)	# Advection coefficient

		f = Expression('100 * (x[0] * x[1] - x[1] * x[1])', degree=2)												# Source term

		u = TrialFunction(V)																						# Trial function
		v = TestFunction(V)																							# Test function
		
		F = sigma * dot(grad(u), grad(v)) * dx + dot(beta, grad(u)) * v * dx - f * v * dx							# Variational formulation
		a, L = lhs(F), rhs(F)

		u = Function(V)																								# Solution function

		solve(a == L, u, bc)																						# Solution of the variational problem

		if plotOn and i % 100 == 0:
			plt.figure(figsize=(4, 3))
			fe.plot(u.vector(), V, colorbar=True)																	# Plot solution
			plt.title('u @ mu = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(mu1t[i], mu2t[i], mu3t[i], mu4t[i]))
			plt.show()
		
		u_test[i] = u.vector()																						# Store snapshot

	print('Saving snapshots...')

	np.save('mu_train.npy', mu_train)
	np.save('mu_test.npy', mu_test)
	np.save('u_train.npy', u_train)
	np.save('u_test.npy', u_test)


# Load snapshots

if not generateData:

	print('Loading snapshots...')

	mu_train = np.load('mu_train.npy')
	mu_test = np.load('mu_test.npy')
	u_train = np.load('u_train.npy')
	u_test = np.load('u_test.npy')


# Traning architecture

m = 16
k = 4

# class Dense(Layer):
# 	 def __init__(self, input_dim, output_dim, activation = leakyReLU):
# 		...

# class Deconv2D(Layer):    
#    def __init__(self, window, channels = (1,1), stride = 1, padding = 0, groups = 1, dilation = 1, activation = leakyReLU):
# 		...
# torch.nn.ConvTranspose2d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

# The input and output dimensions are written in the form height x width x channels (vectors reshaped in 3D tensors)

torch.set_default_dtype(torch.float16)

psi_prime = Dense(Nh, 4)

print(psi_prime.dof())

psi = Dense(4, 100 * m) + \
	  dnns.Deconv2D(11, (4 * m, 2 * m), 2) + \
	  dnns.Deconv2D(10, (2 * m, m), 2) + \
      dnns.Deconv2D(11, (m, 1), 2)

print(psi.dof())

phi = Dense(4, 50 * k) + \
	  Dense(50 * k, 50 * k) + \
	  Dense(50 * k, 4)

print(phi.dof())
