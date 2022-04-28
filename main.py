import math
import numpy as np

# Compute argmin(X) using gradient descent
def compute_min_curvedX(X, Y, D, alpha, tau, num_iterations=10):
    X_old = np.copy(X)

    # Standard gradient descent
    for iters in range(num_iterations):
        grad_X = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                theta_grad = np.dot(X_old[i], Y[j])
                grad_X[i] += Y[j] * (-1 / theta_grad - tau ** 2 * D[i][j] + tau ** 2 * D[i][j] ** 2 * theta_grad) / (
                            X.shape[0] * Y.shape[0])
        a = opt_alpha_X(X_old, Y, grad_X, D, tau)
        X_old -= a * grad_X
        print("Loss(%f): %f" % (a,curved_loss(X_old, Y, D, tau)))

    return X_old, curved_loss(X_old, Y, D, tau)

# Compute loss of curved gaussian
def curved_loss(X, Y, D, tau):
    loss = 0
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            theta = np.dot(X[i], Y[j])
            if theta > 0:
                loss += -math.log(theta)
            loss += - (tau ** 2 * D[i][j] * theta - (tau ** 2 * D[i][j] ** 2 * theta ** 2) / 2)
    return loss / (X.shape[0] * Y.shape[0])

# Initialize dot product model guess
def init_theta(D, iterations=10, tau=2, k=2):
    theta = np.copy(D)
    gamma = 0.001
    for t in range(iterations):
        theta_new = theta - gamma * grad_theta(D, theta, tau)
        X_0, s, Y_0 = np.linalg.svd(theta_new)
        s[k:] = 0
        s_mid = np.zeros(D.shape)
        s_mid[:D.shape[1], :D.shape[1]] = np.diag(s)
        theta = X_0 @ s_mid @ Y_0
    return theta

#Compute optimal gradient step for gene matrix
def opt_alpha_Y(X, Y_old, grad_Y, D, tau):
    alpha_steps = 7
    lb = 0.001
    ub = 1
    base_loss = curved_loss(X, Y_old, D, tau)

    for alpha_i in range(alpha_steps):
        alpha_try = (ub - lb) / 2
        Y_new = Y_old - alpha_try*grad_Y
        new_loss = curved_loss(X, Y_new, D, tau)
        if new_loss < base_loss:
            base_loss = new_loss
            lb = alpha_try
        else:
            ub = alpha_try

    return lb

#Compute optimal gradient step for cell matrix
def opt_alpha_X(X_old, Y, grad_X, D, tau):
    alpha_steps = 6
    lb = 0.001
    ub = 0.5
    base_loss = curved_loss(X_old, Y, D, tau)

    for alpha_i in range(alpha_steps):
        alpha_try = (ub - lb) / 2
        X_new = X_old - alpha_try * grad_X
        new_loss = curved_loss(X_new, Y, D, tau)
        if new_loss < base_loss:
            base_loss = new_loss
            lb = alpha_try
        else:
            ub = alpha_try

    return lb

# Compute gradient of theta
def grad_theta(D, theta, tau):
    g_theta = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            g_theta[i][j] = -1 / theta[i][j] - tau ** 2 * D[i, j] + tau ** 2 * D[i, j] ** 2 * theta[i][j]
    return g_theta / (theta.shape[0] * theta.shape[1])

# Compute argmin(Y) using gradient descent
def compute_min_curvedY(X, Y, D, alpha, tau, num_iterations=10):
    Y_old = np.copy(Y)
    for iters in range(num_iterations):
        grad_Y = np.zeros(Y.shape)
        for j in range(Y.shape[0]):
            for i in range(X.shape[0]):
                theta = np.dot(X[i], Y_old[j])
                grad_Y[j] += X[i] * (-1 / theta - tau ** 2 * D[i][j] + tau ** 2 * D[i][j] ** 2 * theta) / (
                            X.shape[0] * Y.shape[0])
        a = opt_alpha_Y(X, Y_old, grad_Y, D, tau)
        Y_old -= a * grad_Y
        print("Loss(%f): %f" % (a,curved_loss(X, Y_old, D, tau)))
    return Y_old, curved_loss(X, Y_old, D, tau)


# Load synthetic dataset
data = np.load("data/synth_Dk2_1.npy")  # dataset[cell][protein]
X_ = np.load("data/synth_Xk2_1.npy")

# Experiment 2: eSVD recreation with synthetic data
n = data.shape[0]
p = data.shape[1]

k = 2   #Latent space dimensions
tau = 2 #Curved Gaussian parameter
T = 10

# Initialize Theta
# For curved gaussian, inverse derivative of the log-partition function is the original matrix
theta = init_theta(data)

X_0, s, Y_0 = np.linalg.svd(theta)
Y_iter = np.copy(Y_0).T[:, :k]
Y_bar = np.copy(Y_0).T[:, :k]
X_iter = (np.copy(X_0))[:, :k]
X_bar = (np.copy(X_0))[:, :k]

loss = np.zeros([T, 2])

# eSVD Implementation
for t in range(T):
    # Compute argmin(X) for curved gaussian
    X_iter, loss[t, 0] = compute_min_curvedX(X_bar, Y_bar, data, alpha=0.001, tau=2, num_iterations=1000)
    X_0, s, Y_0 = np.linalg.svd(X_iter)
    X_bar = math.sqrt(n) * np.copy(X_0)[:, :k]

    # Compute argmin(Y) for curved gaussian
    Y_iter, loss[t, 1] = compute_min_curvedY(X_bar, Y_bar, data, alpha=0.001, tau=2, num_iterations=1000)
    X_0, s, Y_0 = np.linalg.svd(Y_iter)
    Y_bar = math.sqrt(p) * np.copy(X_0)[:, :k]

# Final computation for eSVD to generate final X and Y
theta = X_bar @ Y_iter.T
X_0, s, Y_0 = np.linalg.svd(theta)
s[k:] = 0
s_mid = np.zeros(data.shape)
s_mid[:len(s), :len(s)] = np.diag(s)

X_hat = X_0 @ np.sqrt(s_mid) * (n / p) ** (1 / 4)
Y_hat = np.sqrt(s_mid) @ Y_0 * (p / n) ** (1 / 4)

# Save reconstructed data. Purposefully different dataset from those saved.
np.save("data/synth_loss_eSVDk2_theta4", theta)
np.save("data/synth_loss_eSVDk2_curved_set4", loss)
np.save("data/synth_XeSVDk2_curved_set4", X_hat)
np.save("data/synth_YeSVDk2_curved_set4", Y_hat)
