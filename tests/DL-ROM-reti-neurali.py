from dlroms import *

### ESEMPIO 1

from IPython.display import clear_output as clc
from dlroms.gp import GaussianRandomField
import numpy as np

domain = fe.circle((0,0), 1)
mesh = fe.mesh(domain, resolution = 40)
V = fe.space(mesh, 'CG', 1)

G = GaussianRandomField(mesh, kernel = lambda r: np.exp(-4*r**2), upto = 30)
clc()

n = 1000 # numero di dati
u, y = [], [] # liste vuote

for i in range(n):
  ui = G.sample(i)
  u.append(ui)
  y.append(np.mean(ui)) # approssimiamo la media integrale con una media sui nodi

# Da liste ad array
u = np.stack(u)
y = np.stack(y).reshape(n, 1)

# Ora u è un array n x nh, poiché contiene n oggetti dello spazio V, dim(V) = nh
# Ora y è un array n x 1, in quanto abbiamo n diversi output scalari

# Trasferimento su Pytorch (da numpy array a tensori torch)
u = CPU.tensor(u)
y = CPU.tensor(y)

# Costruiamo una rete mesh-informed + dense, per gestire l'input "funzionale"

# Mesh coarse ausiliaria
mesh_coarse = fe.mesh(domain, resolution = 10)
V_coarse = fe.space(mesh_coarse, 'CG', 1)

nh = V.dim()
n_coarse = V_coarse.dim()

print("Dof mesh fine: %d. Dof mesh coarse: %d." % (nh, n_coarse))

# Architettura: rete a 3 layer, che parte da V e mappa in R (dim = 1).
rete = Local(V, V_coarse, support = 0.1) + Dense(n_coarse, 10) + Dense(10, 1, activation = None)

# NB: è lecita anche la notazione
# rete = Local(V, V_coarse, support = 0.1) + Dense(V_coarse, 10) + Dense(10, 1, activation = None)

# Inoltre, la chiamata "Local(A, B, support)" si può effettuare anche con A e B liste di punti (es: coordinate dei dof degli spazi)

print("Parametri trainabili della rete (pesi e bias): %d" % rete.dof())

# NB: se U ha shape m x nh, rete(U) avrà shape m x 1. In generale, le reti operano su tutte le dimensioni fuorché la prima, che rappresenta invece la "dimensione di batch"
# (per m input, vorrei m output: si lavora quindi in parallelo).

rete = DFNN(rete) # converte l'architettura in un oggetto trainabile (non necessario in torch ma utile in dlroms)

# La loss function è ciò che la rete cerca di minimizzare durante il training.
# Le seguenti definizioni sono tutte EQUIVALENTI:

# MODO 1
def loss(vero, predetto):
  return (vero-predetto).pow(2).sum(axis = -1).mean()

# la prima somma, "sum(axis = -1)" serve ad accorpare gli errori presenti nelle componenti del singolo output (anche se qui c'è una sola componente...).
# "vero" e "predetto" avranno shape (n, 1), perché sono n output 1-dimensionali
# (vero-predetto).pow(2) ha ancora shape (n, 1)
# (vero-predetto).pow(2).sum(axis = -1) ha shape (n,), perché abbiamo sommato sulle componenti (qui una sola)
# (vero-predetto).pow(2).sum(axis = -1).mean() è la media del precedente su "n"

# MODO 2
loss = lambda vero, predetto: (vero-predetto).pow(2).sum(axis = -1).mean()

# MODO 3
loss = mse(euclidean) # mse(norm) genere una lambda function della forma: vero, predetto -> mean(norm(vero-predetto)**2)

# E' utile avere anche una funzione d'errore (o di "performance"), che ci dica quanto sono buone le predizioni.
# Qui usiamo

def error(vero, predetto):
  return (vero-predetto).abs().sum(axis = -1).mean() # NB: come prima, la somma è fittizia, serve solo a togliere la dimensione "scalare"

# Training: usiamo il 60% dei dati per fare training, il 20% come insieme di validazione, ed il restante 20% come testing.

# Dati di training = quelli su cui si calcola la loss e si minimizza.
# Dati di validation = dati che vengono monitorati: se la performance migliora sui dati di training ma peggiora su quelli di validazione, l'allenamento viene interrotto.
# Dati di testi = quelli su cui effettivamente valutare (alla fine) la bontà del modello.

# NB: non è obbligatorio avere un validation set. Se non si passa l'argomento "nvalid", quest'ultimo verrà ignorato.
ntrain = int(0.6*n)
nvalid = int(0.2*n)

rete.He() # inizializzazione random di pesi e bias
rete.train(u, y, ntrain = ntrain, nvalid = nvalid, epochs = 50, loss = loss, error = error)

# Volendo si possono cambiare ottimizzatore e learning rate, es: rete.train(..., optim = torch.optim.Adam, lr = 1e-3)
# o anche introdurre strategie di minibatching.

unew = G.sample(40001)

print("Valore vero G(u): %.3f" % np.mean(unew))
print("Valore predetto:  %.3f" % rete(CPU.tensor(unew).reshape(1, -1))[0])

# Dopo il training, si può risparmiare memoria congelando la rete

rete.freeze()

# Per valutare singoli output, si può anche usare la chiamata 'solve':
print("Valore vero G(u): %.3f" % np.mean(unew))
print("Valore predetto:  %.3f" % rete.solve(unew))

### ESEMPIO 2

print(u.shape) # shape dei dati di input

v = u/(1+u.abs()) # output (definito così, è già un tensore torch contenente n realizzazioni di dimensione nh)

plt.figure(figsize = (6, 3))
plt.subplot(1,2,1)
fe.plot(u[0], V)
plt.title("Funzione in input")
plt.subplot(1,2,2)
fe.plot(v[0], V)
plt.title("Funzione in output")
plt.show()

# Architettura: rete a 3 layer da V in V
dnn = Local(V, V, support = 0.1) + Local(V, V, support = 0.1) + Local(V, V, support = 0.1, activation = None)
dnn = DFNN(dnn)
dnn.He()

# Ora per la loss è più opportuno usare una norma funzionale, es: la norma L2. Per fare ciò basta
# costruire l'oggetto "norma l2":

l2 = L2(V) # calcola norme L2 di famiglie di funzioni definite su V.
clc()

# Ad es: se u ha shape n x nh (cioè n funzioni definite su V), allora l2(u) ha shape n x 1 (cioè n norme l2). Come le reti, anche questo oggetto lavora in batches.

loss = mse(l2)
error = mre(l2) # Mean Relative Error. Definisce una lambda function: vero, predetto -> mean( norma(vero-predetto) / norma(vero) )

dnn.He()
dnn.train(u, v, ntrain = ntrain, nvalid = nvalid, epochs = 15, loss = loss, error = error, notation = '%')

dnn.freeze()

# A titolo d'esempio, vediamo gli output proposti per una delle istanze presenti nel test set

plt.figure(figsize = (9, 3))
plt.subplot(1,3,1)
fe.plot(u[-1], V)
plt.title("Input")
plt.subplot(1,3,2)
fe.plot(v[-1], V)
plt.title("Output")
plt.subplot(1,3,3)
fe.plot(dnn.solve(u[-1]), V)
plt.title("Approssimazione")
plt.show()
