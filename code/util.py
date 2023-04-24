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
        plt.colorbar()
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
        neighbors = X[indices]
        point = X[i].reshape(1,-1)
        dX = point - neighbors # shape (n_neighbors, n_features)
        dX /= l2_norm(dX)[:, None]

        # dX /= np.sqrt(dX.multiply(dX).sum(axis=1).A1)[:, None]
        dX[np.isnan(dX)] = 0  # zero diff in a steady-state
        #neighbor edge weights are used to weight the overall dX or velocity from cell i.
        probs =  T[i].data
        # TODO what is the second term doing?
        V[i] = probs.dot(dX) #- probs.mean() * dX.sum(0)
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
    plt.title(f'{genotype.capitalize()} Distribution of total gene expression per cell across all genes');
    plt.savefig(f'{figdir}/{name}_total_expression_per_cell_{genotype}.png', dpi=300)
    plt.close()

    # Plot the distribution of gene expression for each gene
    plt.hist(np.log10(adata.X.sum(axis=0)+1), bins=100)
    plt.title(f'{genotype.capitalize()} Log Distribution of total expression per gene across all cells');
    plt.savefig(f'{figdir}/{name}_log_expression_per_gene_{genotype}.png', dpi=300)
    plt.close()

    # Plot the number of genes with expression > 0 per cell
    plt.hist((adata.X>0).sum(axis=0), bins=100);
    plt.title(f'{genotype.capitalize()} Distribution of number of cells with expression > 0 per gene');
    plt.savefig(f'{figdir}/{name}_nonzero_expression_per_gene_{genotype}.png', dpi=300)
    plt.close()

    # Plot the cumulative distribution of total gene expression per cell
    plt.hist(adata.X.sum(axis=1), bins=100, cumulative=True);
    plt.title(f'{genotype.capitalize()} Cumulative distribution of total gene expression per cell');
    plt.savefig(f'{figdir}/{name}_cumulative_expression_per_cell_{genotype}.png', dpi=300)
    plt.close()

import pickle
def filter_to_network(adata):    
    # Import gene network from Tiana et al paper
    graph = pickle.load(open('../data/filtered_graph.pickle', 'rb'))
    protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
    protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))
    indices_of_nodes_in_graph = []
    data_ids = {}
    id_new_row = {}
    new_row = 0
    for i,name in enumerate(adata.var_names):
        name = name.upper()
        if name in protein_name_to_ids:
            for id in protein_name_to_ids[name]:
                if id in graph.nodes:
                    indices_of_nodes_in_graph.append(i)
                    if id in data_ids:
                        print('Duplicate id', id, name, data_ids[id])
                    data_ids[id] = name
                    id_new_row[id] = new_row
                    new_row += 1
    # Filter the data to only include the genes in the Nanog regulatory network
    network_data = adata[:,indices_of_nodes_in_graph]
    network_data.var_names = [adata.var_names[i] for i in indices_of_nodes_in_graph]
    network_data.uns['id_row'] = id_new_row
    return network_data


def umap_axes(axs):
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

def get_plot_limits(X, buffer=0.1):
    # Get the x and y extents of the embedding
    x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
    y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    x_buffer = x_diff * buffer
    y_buffer = y_diff * buffer
    x_limits = (x_min-x_buffer, x_max+x_buffer)
    y_limits = (y_min-y_buffer, y_max+y_buffer)
    return x_limits, y_limits
