from dlroms import *

from IPython.display import clear_output as clc

domain = fe.circle((0,0), 0.75) - fe.circle((0.0, 0.0), 0.4) # dominio spaziale, ottenuto come differenza di due cerchi
mesh = fe.mesh(domain, resolution = 30) # costruzione della mesh
V = fe.space(mesh, 'CG', 1) # spazio elementi finiti corrispondente (P1)
clc()

plt.figure(figsize = (3, 3))
fe.plot(mesh)
plt.show()

x, y = fe.coordinates(V).T # coordinate dei nodi rappresentati i dof di V. Se nh = dim(V), x ed y sono vettori lunghi nh

u = x**2 + y**2 # essendo una base nodale, questo vettore corrisponderà alla versione "dof" della mappa x^2 + y^2

# Un numpy array (o un tensore torch) di dimensione nh = dim(V) è pensato come la versione discreta di una funzione definita sul dominio
# La libreria dlroms.fespaces permette di: i) passare dalla rappresentazione vettoriale a quella funzionale (raramente utile)
#                                          ii) plottare (molto più utile)

u_function = fe.asvector(u, V) # lo rende un "vettore" dello spazio elementi finiti (pertanto una funzione)
print(u_function(0.5, 0.5)) # valuta nel punto

plt.figure(figsize = (3, 3)) # la gestione generica dei plot resta in matplotlib.pyplot
fe.plot(u, V, colorbar = True, shrink = 0.5) # si passa la versione discreta, non quella funzionale!
plt.show()

from dlroms.gp import GaussianRandomField
import numpy as np

# Campo gaussiano classico
G = GaussianRandomField(mesh, kernel = lambda r: np.exp(-4*r**2), upto = 30) # kernel di covarianza Cov(xi, xj) = exp(-4|xi-xj|^2)
clc()

# Per ottenere una realizzazione casuale, basta fare G.sample(seed)
u0 = G.sample(5) # sarà un numpy array di lunghezza nh = dim(V)

plt.figure(figsize = (3, 3))
fe.plot(u0, V)
plt.show()

# Campo gaussiano con kernel basato sulla distanza geodesica (approssimata)
G1 = GaussianRandomField(mesh, kernel = lambda r: np.exp(-4*r**2), upto = 30, domain = domain, geodesic_accuracy = 20) # kernel Cov(xi, xj) = exp(-4d(xi, xj)^2)
clc()

u1 = G1.sample(5)
plt.figure(figsize = (3, 3))
fe.plot(u1, V)
plt.show()

# NB: è ovviamente meno smooth perché la distanza geodesica è più "spigolosa" di quella Euclidea

brain = np.load("./brainshape.npz")
domain = fe.polygon(brain["main"][::9]) - fe.polygon(brain["hole1"][::9]) - fe.polygon(brain["hole2"][::8])

plt.figure(figsize = (4, 4))
fe.plot(fe.mesh(domain, resolution = 20))
plt.show()
