import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from sklearn.cluster import KMeans


#Computes M-Step for EM
def EM_M(Y, r, pi, a, sigma):

    m = np.sum(r, axis=0)
    pi = m/sum(m)
    r = r[:, :, np.newaxis, np.newaxis]

    x = Y[:, np.newaxis, :, np.newaxis]
    x_T = np.transpose(x, axes=[0, 1, 3, 2])
    #Gradient descent to compute optimal sigma and a
    lr = 0.001
    epochs = 10

    #Gradient descent for maximization step
    for epoch in range(epochs):
        sigma_t = np.transpose(sigma[np.newaxis, :], axes=[0, 1, 3, 2]) + sigma[np.newaxis, :]
        del_a = a[np.newaxis, :, 1, np.newaxis] - a[np.newaxis, :, 0, np.newaxis]
        del_a = np.squeeze(del_a, axis=2)
        del_a0 = (1 / 3)*(del_a[:,:,:,np.newaxis]) - 0.5*(x)
        grad_a0 = np.squeeze(np.sum(r*tf.matmul(sigma_t, del_a0), axis=0),2) / Y.shape[0]
        del_a1 = -(1 / 3)*(del_a[:,:,:,np.newaxis]) + 0.5*(x)
        grad_a1 = np.squeeze(np.sum(r*tf.matmul(sigma_t, del_a1), axis=0),2) / Y.shape[0]

        a[:, 0] += -lr * grad_a0
        a[:, 1] += -lr * grad_a1

        del_a = a[np.newaxis, :, 1, np.newaxis] - a[np.newaxis, :, 0, np.newaxis]
        del_a = np.squeeze(del_a, axis=2)
        mu = del_a[:,:,:,np.newaxis]
        mu_T = np.transpose(del_a[:,:,:,np.newaxis], axes=[0, 1, 3, 2])
        d_sig1 = np.transpose(sigma[np.newaxis, :], axes=[0, 1, 3, 2])
        d_sig2 = tf.matmul(x,x_T)
        d_sig3 = tf.matmul(mu,mu_T)
        d_sig4 = tf.matmul(x,mu_T)
        d_sig5 = tf.matmul(mu,x_T)

        del_sigma = -.5 * ( d_sig1 + d_sig2 + (1/3) * d_sig3 + .5 * (d_sig4 + d_sig5) )
        grad_sigma = np.sum(r*del_sigma, axis=0) / Y.shape[0]

        sigma -= lr * grad_sigma

    return pi, a, sigma

#Computes E-Step for EM
def EM_E(x, a, sigma, pi):
    x_T = np.repeat(np.expand_dims(x, axis=(1, 2)), 3, axis=1)
    sigma_exp = np.repeat(np.expand_dims(sigma, axis=0), x.shape[0], axis=0)
    x = np.repeat(np.expand_dims(x, axis=(1, 3)), 3, axis=1)
    del_a = a[np.newaxis, :, 1, np.newaxis] - a[np.newaxis, :, 0, np.newaxis]
    del_a = np.squeeze(del_a, axis=2)
    mu = x - del_a[:,:,:,np.newaxis]
    mu_T = x_T - (a[np.newaxis, :, 1, np.newaxis] - a[np.newaxis, :, 0, np.newaxis])

    t_1 = tf.matmul(tf.matmul(x_T, sigma_exp), x)
    t_2 = tf.matmul(tf.matmul(mu_T, sigma_exp), x)
    t_3 = tf.matmul(tf.matmul(x_T, sigma_exp), mu)
    t_4 = tf.matmul(tf.matmul(mu_T, sigma_exp), mu)

    sum_t = -.5*(t_1 - t_2/2 - t_3/2 + t_4/3)
    y = np.exp(np.squeeze(sum_t))
    y[:, 0] /= np.math.sqrt(2*math.pi*np.linalg.det(sigma[0]))
    y[:, 1] /= np.math.sqrt(2 * math.pi * np.linalg.det(sigma[1]))
    y[:, 2] /= np.math.sqrt(2 * math.pi * np.linalg.det(sigma[2]))

    y = y * pi[np.newaxis,:]
    r = y / np.sum(y,axis=1)[:,np.newaxis]


    return r

#Computes full EM
def EM():
    X = np.load("data/synth_XeSVDk2_curved_set3.npy")

    # Remove outliers to make the figure look prettier
    X_fil = X[:, :2]
    x_index = np.all(abs(X_fil - np.mean(X_fil, axis=0)) < 2 * np.std(X_fil, axis=0), axis=1)
    X = X[x_index]
    Y = X[:, :2]

    num_data = Y.shape[0]

    a_init = np.array([[[0, 0],[-.9, .9]],[[0, 0],[-1.5, 0]],[[0, 0],[-1, -.75]]])
    a = np.copy(a_init)
    pi = np.array([1/3, 1/3, 1/3])
    sigma = np.zeros([3, 2, 2])
    sigma[0, :] = np.array([[1, 0], [0, 1]])
    sigma[1, :] = np.array([[1, 0], [0, 1]])
    sigma[2, :] = np.array([[1, 0], [0, 1]])
    sigma *= 0.25

    r = EM_E(Y, a, sigma, pi)
    pi, a, sigma = EM_M(Y, r, pi, a, sigma)

    iter_t = 100
    loss = np.zeros(iter_t)
    for t in range(iter_t):
        r = EM_E(Y, a, sigma, pi)
        pi, a, sigma = EM_M(Y, r, pi, a, sigma)


    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axes.set_ylabel('Latent dimension 2')
    axes.set_xlabel('Latent dimension 1')
    axes.set_title('eSVD Synthetic Lineages')

    axes.scatter(Y[:, 0], Y[:, 1])
    axes.plot(a[0, :, 0], a[0, :, 1], label="Line 1: EM Fit",c='red')
    axes.plot(a[1, :, 0], a[1, :, 1], label="Line 2: EM Fit",c='black')
    axes.plot(a[2, :, 0], a[2, :, 1], label="Line 3: EM Fit", c='green')

    #axes.plot(a_init[0, :, 0], a_init[0, :, 1], label="Line 1: Initial")
    #axes.plot(a_init[1, :, 0], a_init[1, :, 1], label="Line 2: Initial")
    #axes.plot(a_init[2, :, 0], a_init[2, :, 1], label="Line 3: Initial")

    plt.legend()

    plt.show()

    return loss



EM()





