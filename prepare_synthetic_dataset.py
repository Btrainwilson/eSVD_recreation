import numpy as np
import random

# Hyper-parameters
n = 1000    # Number of samples
k = 2       # Latent space dimension
p = 20      # Number of genes

# Cell latent space matrix
X = np.zeros([n, 3])  # X[cell] = [x_1, x_2, cluster_id]

# Generate n samples
for i in range(n):

    # Choose curve class [0,1,2,3]
    cluster = random.randint(0, 3)

    # Uniformly choose parameter index along curve
    param = random.uniform(0, 1)

    X[i, :2] = [random.gauss(0.1, 0.025), random.gauss(0.1, 0.025)]
    X[i, 2] = cluster

    if cluster == 0:
        X[i, 1] += param

    elif cluster == 1:
        X[i, 0] += param * .5

    elif cluster == 2:
        X[i, 0] += param * .5
        X[i, 1] += 0.5

    elif cluster == 3:
        X[i, 0] += param * .5
        X[i, 1] += 1

# Gene latent space matrix
Y = np.zeros([n, 2])  # Y[cell] = [y_1, y_2]
for i in range(p):
    Y[i, :] = [random.gauss(0.5, 0.02), random.gauss(0.5, 0.02)]

# Curve parameter for curved gaussian
tau = 2

# Construct data matrix for curved Gaussian parameterized by the dot product model (X^T)Y
D = np.zeros([n, p])
for i in range(n):
    for j in range(p):
        mu = 1 / np.dot(X[i, :2], Y[j])
        D[i, j] = random.gauss(mu, mu ** 2 / tau ** 2)

# Save matrices
np.save("data/synth_Yk2_1", Y)
np.save("data/synth_Xk2_1", X)
np.save("data/synth_Dk2_1", D)
