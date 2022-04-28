import numpy as np
import matplotlib.pyplot as plt
from slingshot import Slingshot

def closest_point_index(arr, x, y):
    dist = (arr[:, 0] - x)**2 + (arr[:, 1] - y)**2
    return dist.argmin()

# Figure 1: Synthetic Latent Space (Not
def figure_1(X):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
    plt.ylabel("Latent dimension 2")
    plt.xlabel("Latent dimension 1")
    plt.title("Synthetic Latent Trajectory")
    plt.savefig("figures/fig_1_synthetic_trajectories")

# figures_original_slingshot: Slingshot on original synthetic data. Generates part a) for all figures
def figures_original_slingshot():

    # Load latent representation that generated synthetic data
    X = np.load("data/synth_Xk2_1.npy")

    # Compute cluster labels and one-hot encoding of cluster labels for Slingshot
    cluster_labels = X[:, 2].astype(dtype=int)
    cluster_labels_onehot = np.zeros((X.shape[0], cluster_labels.max()+1))
    cluster_labels_onehot[np.arange(X.shape[0]), cluster_labels] = 1

    # Setup Figures for Slingshot Debug Plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes[0, 1].set_ylabel('Latent dimension 2')
    axes[0, 1].set_xlabel('Latent dimension 1')
    axes[0, 1].set_title('Ideal Synthetic Lineages')

    # Run Slingshot
    slingshot = Slingshot(X[:, :2], cluster_labels_onehot, start_node=0, debug_level='verbose')
    slingshot.fit(num_epochs=10, debug_axes=axes)

    # Setup figures for plotting principle curves
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    axes[0].set_title('Ideal Synthetic Clusters')
    axes[0].set_ylabel('Latent dimension 2')
    axes[1].set_ylabel('Latent dimension 2')
    axes[0].set_xlabel('Latent dimension 1')
    axes[1].set_xlabel('Latent dimension 1')
    axes[1].set_title('Ideal Synthetic Pseudotime')

    # Plot principle curves and pseudotime from Slingshot
    slingshot.plotter.curves(axes[0], slingshot.curves)
    slingshot.plotter.clusters(axes[0], labels=[r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$"], s=4, alpha=0.5)
    slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)

    plt.show()

# figures_reconstructed_slingshot: Slingshot on reconstructed synthetic data. Generates part b) for all figures
def figures_reconstructed_slingshot():

    # Load latent representation computed with eSVD
    X = np.load("data/synth_XeSVDk2_curved_set3.npy")

    #Remove outliers to make the figure look prettier
    X_fil = X[:, :2]
    x_index = np.all(abs(X_fil - np.mean(X_fil, axis=0)) < 2 * np.std(X_fil, axis=0), axis=1)
    X = X[x_index]

    # Load cluster labels for all latent space points
    X_l = np.load("data/synth_Xk2_1.npy")

    # Compute cluster labels and one-hot encoding of cluster labels for Slingshot
    cluster_labels = X_l[x_index, 2].astype(dtype=int)
    cluster_labels_onehot = np.zeros((X.shape[0], cluster_labels.max()+1))
    cluster_labels_onehot[np.arange(X.shape[0]), cluster_labels] = 1

    # Setup Figures for Slingshot Debug Plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes[0, 1].set_ylabel('Latent dimension 2')
    axes[0, 1].set_xlabel('Latent dimension 1')
    axes[0, 1].set_title('eSVD Synthetic Lineages')

    # Run Slingshot
    slingshot = Slingshot(X[:, :2], cluster_labels_onehot, start_node=0, debug_level='verbose')
    slingshot.fit(num_epochs=10, debug_axes=axes)

    # Setup figures for plotting principle curves
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    axes[0].set_title('eSVD Synthetic Clusters')
    axes[0].set_ylabel('Latent dimension 2')
    axes[1].set_ylabel('Latent dimension 2')
    axes[0].set_xlabel('Latent dimension 1')
    axes[1].set_xlabel('Latent dimension 1')
    axes[1].set_title('eSVD Synthetic Pseudotime')

    # Plot principle curves and pseudotime from Slingshot
    slingshot.plotter.curves(axes[0], slingshot.curves)
    slingshot.plotter.clusters(axes[0], labels=[r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$"], s=4, alpha=0.5)
    slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)

    plt.show()

def figures_reconstructed_no_slingshot():

    # Load latent representation computed with eSVD
    X = np.load("data/synth_XeSVDk2_curved_set3.npy")[:, :2]

    # Load cluster labels for all latent space points
    X_l = np.load("data/synth_Xk2_1.npy")

    # Compute cluster labels and one-hot encoding of cluster labels for Slingshot
    cluster_labels = X_l[:, 2].astype(dtype=int)
    cluster_labels_onehot = np.zeros((X.shape[0], cluster_labels.max()+1))
    cluster_labels_onehot[np.arange(X.shape[0]), cluster_labels] = 1

    # Setup figures for plotting principle curves
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    axes[0].set_title('eSVD Synthetic Clusters')
    axes[0].set_ylabel('Latent dimension 2')
    axes[1].set_ylabel('Latent dimension 2')
    axes[0].set_xlabel('Latent dimension 1')
    axes[1].set_xlabel('Latent dimension 1')
    axes[1].set_title('eSVD Synthetic Pseudotime')

    x_index = np.all(abs(X - np.mean(X, axis=0)) < 2 * np.std(X, axis=0), axis=1)
    X_filtered = X[x_index]

    axes[0].scatter(X_filtered[:, 0], X_filtered[:, 1])

    plt.show()

#figures_original_slingshot()
figures_reconstructed_slingshot()
#figures_reconstructed_no_slingshot()