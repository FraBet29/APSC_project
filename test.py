import numpy as np
from dlroms import *
import numpy.random as rnd
import matplotlib.pyplot as plt
import torch
from dolfin import *


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


# Snapshots generation

print('Generating snapshots...')

for i in range(training_size):

	if i % 100 == 0:
		print('Snapshot {}/{}'.format(i, training_size))
	
	sigma = Expression('6 + 5 * tanh(20 * (x[1] + 10 * mu1 * x[0] * (x[0] - 1) * (x[0] - mu2) * (x[0] - mu3) - 0.5))', 
				   		degree=2, mu1=mu1[i], mu2=mu2[i], mu3=mu3[i])											# Diffusion coefficient
	
	if i % 100 == 0: 
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

	if i % 100 == 0:
		plt.figure(figsize=(4, 3))
		fe.plot(u.vector(), V, colorbar=True)																	# Plot solution
		plt.title('u @ mu = ({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(mu1[i], mu2[i], mu3[i], mu4[i]))
		plt.show()





"""
# Traning architecture

m = 16
k = 4

# class Dense(Layer):
# 	 def __init__(self, input_dim, output_dim, activation = leakyReLU):
# 		...

# class Deconv2D(Layer):    
#    def __init__(self, window, channels = (1,1), stride = 1, padding = 0, groups = 1, dilation = 1, activation = leakyReLU):
# 		...

torch.set_default_dtype(torch.float16)

psi_prime = Dense(Nh, 4)

print(psi_prime.dof())

psi = Dense(4, 100 * m) + \
	  dnns.Deconv2D(11, (5 * 5 * 4 * m, 19 * 19 * 2 * m), 2) + \
	  dnns.Deconv2D(10, (19 * 19 * 2 * m, 46 * 46 * m), 2) + \
      Deconv2D(11, (47 * 47 * m, 101 * 101), 2)

print(psi.dof())

phi = Dense(4, 50 * k) + \
	  Dense(50 * k, 50 * k) + \
	  Dense(50 * k, 4, activation = 'None')

print(phi.dof())
"""