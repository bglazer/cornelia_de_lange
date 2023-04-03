import numpy as np
from matplotlib import pyplot as plt

def tonp(x):
    return x.detach().cpu().numpy()

from matplotlib import pyplot as plt
def plot_arrows(idxs, points, V, pV=None, sample_every=10, scatter=True, save_file=None, c=None, s=3, aw=0.001, xlimits=None, ylimits=None, alphas=1):
    # Plot the vectors from the sampled points to the transition points
    plt.figure(figsize=(15,15))
    if scatter:
        plt.scatter(points[:,0], points[:,1], s=s, c=c)
    plt.xlim=xlimits
    plt.ylim=ylimits
    # black = true vectors
    # Green = predicted vectors
    sample = points[idxs]
    
    for i in range(0, len(idxs), sample_every):
        if type(alphas) == int:
            alpha = alphas
        else:
            alpha = alphas[i]
        plt.arrow(sample[i,0], sample[i,1], V[i,0], V[i,1], color='black', alpha=alpha, width=aw)
        if pV is not None:
            plt.arrow(sample[i,0], sample[i,1], pV[i,0], pV[i,1], color='g', alpha=alpha, width=aw)

    # Remove the ticks and tick labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # Add a legend in the top right corner labeling the arrow colors
    plt.legend(['True', 'Predicted'], loc='upper right')
    
    if save_file is not None:
        plt.savefig(save_file)

def l2_norm(x):
    return np.sqrt((x ** 2).sum(axis=1))

#the change in embedding distance when moving from cell i to its neighbors is given by dx
def velocity_vectors(T, X):
    V = np.zeros(X.shape)
    n_obs = X.shape[0]
    for i in range(n_obs):
        indices = T[i].indices
        dX = X[indices] - X[i, None]  # shape (n_neighbors, 2)
        dX /= l2_norm(dX)[:, None]

        # dX /= np.sqrt(dX.multiply(dX).sum(axis=1).A1)[:, None]
        dX[np.isnan(dX)] = 0  # zero diff in a steady-state
        #neighbor edge weights are used to weight the overall dX or velocity from cell i.
        probs =  T[i].data
        #if probs.size ==0: print('velocity embedding probs=0 length', probs, i, self.true_label[i])
        V[i] = probs.dot(dX) - probs.mean() * dX.sum(0)
    # Set rows with any nan to all zero
    V[np.isnan(V).sum(axis=1) > 0] = 0
    return V

def embed_velocity(X, velocity, embed_fn):
    dX = X + velocity
    V_emb = embed_fn(dX)
    X_embedding = embed_fn(X)
    dX_embed = X_embedding - V_emb

    return dX_embed

def plot_qc_distributions(adata, genotype, name, figdir):
    # Plot the overall distribution of total gene expression
    plt.hist(adata.X.sum(axis=1), bins=100)
    plt.title('Distribution of total gene expression per cell across all genes');
    plt.savefig(f'{figdir}/{name}_total_expression_per_cell_{genotype}.png', dpi=300)
    plt.close()

    # Plot the distribution of gene expression for each gene
    plt.hist(np.log10(adata.X.sum(axis=0)+1), bins=100)
    plt.title('Log Distribution of total expression per gene across all cells');
    plt.savefig(f'{figdir}/{name}_log_expression_per_gene_{genotype}.png', dpi=300)
    plt.close()

    # Plot the number of genes with expression > 0 per cell
    plt.hist((adata.X>0).sum(axis=0), bins=100);
    plt.title('Distribution of number of cells with expression > 0 per gene');
    plt.savefig(f'{figdir}/{name}_nonzero_expression_per_gene_{genotype}.png', dpi=300)
    plt.close()

    # Plot the cumulative distribution of total gene expression per cell
    plt.hist(adata.X.sum(axis=1), bins=100, cumulative=True);
    plt.title('Cumulative distribution of total gene expression per cell');
    plt.savefig(f'{figdir}/{name}_cumulative_expression_per_cell_{genotype}.png', dpi=300)
    plt.close()

