#%%
import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
from pyVIA.core import l2_norm
from flow_model import FlowModel

#%%
genotype='wildtype'

#%% 
# Load the VIA object from pickle
via = pickle.load(open(f'../data/{genotype}_pseudotime.pickle', 'rb'))

#%%
# Load the umap embedding
# embedding = pickle.load(open(f'../data/umap_embedding_{genotype}.pickle', 'rb'))
# Load the umap object
umap_ = pickle.load(open(f'../data/umap_{genotype}.pickle', 'rb'))
# Load the pca object
pca_ = pickle.load(open(f'../data/pca_{genotype}.pickle', 'rb'))
# Load the graph object
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))

#%%
# Get the transition matrix from the VIA graph
X = via.data
T = via.sc_transition_matrix(smooth_transition=1)
V = np.zeros(X.shape)
n_obs = X.shape[0]
#the change in embedding distance when moving from cell i to its neighbors is given by dx
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

# bad hack that I have to do because VIA doesn't like working with low dimensional data
# Setting nan values to zero. Nan values occur when a point has no neighbors
V[np.isnan(V).sum(axis=1) > 0] = 0

#%%
def tonp(x):
    return x.detach().cpu().numpy()

#%%
def plot_arrows(idxs, points, V, pV=None, sample_every=10, scatter=True, save_file=None, c=None, s=3, aw=0.001, xlimits=None, ylimits=None):
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
        plt.arrow(sample[i,0], sample[i,1], V[i,0], V[i,1], color='black', alpha=1, width=aw)
        if pV is not None:
            plt.arrow(sample[i,0], sample[i,1], pV[i,0], pV[i,1], color='g', alpha=1, width=aw)

    # Remove the ticks and tick labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    if save_file is not None:
        plt.savefig(save_file)

#%%
def embed(X):
    return pca_.transform(X)[:,1:3]

#%%
def embed_velocity(X, velocity, embed_fn):
    dX = X + velocity
    V_emb = embed_fn(dX)
    X_embedding = embed_fn(X)
    dX_embed = X_embedding - V_emb

    return dX_embed


#%%
embedding = embed(via.data)
V_emb = embed_velocity(via.data, V, embed)

#%%
# Get the x and y extents of the embedding
x_min, x_max = np.min(embedding[:,0]), np.max(embedding[:,0])
y_min, y_max = np.min(embedding[:,1]), np.max(embedding[:,1])
x_diff = x_max - x_min
y_diff = y_max - y_min
x_buffer = x_diff * 0.1
y_buffer = y_diff * 0.1
x_limits = (x_min-x_buffer, x_max+x_buffer)
y_limits = (y_min-y_buffer, y_max+y_buffer)

#%%
# plot_arrows(idxs=range(len(embedding)), 
#             points=embedding, 
#             V=V_emb, 
#             sample_every=2, 
#             c=via.single_cell_pt_markov,
#             xlimits=x_limits,
#             ylimits=y_limits)

#%%
num_layers = 3

#%%
network_data = pickle.load(open(f'../data/network_data_{genotype}.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))

nodes = list(network_data.var_names)
indices_of_nodes_in_graph = []
node_idxs = {}
# Get the indexes of the data rows that correspond to nodes in the Nanog regulatory network
for i,name in enumerate(nodes):
    name = name.upper()
    # Find the ensembl id of the gene
    if name in protein_name_to_ids:
        # There may be multiple ensembl ids for a gene name
        for id in protein_name_to_ids[name]:
            # If the ensembl id is in the graph, then the gene is in the network
            if id in graph.nodes:
                # Record the data index of the gene in the network data
                node_idxs[id] = i

#%%
device = 'cuda:0'
model = FlowModel(num_layers=num_layers, 
                  graph=graph, 
                  data_idxs=node_idxs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')

data = torch.tensor(via.data).to(torch.float32).to(device)
Vgpu = torch.tensor(V).to(torch.float32).to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 10000
n_points = 1000
n_traces = 50
n_samples = 10

losses = {node : [] for node in graph.nodes}
total_losses = []

#%%
for i in range(n_epoch+1):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    idxs = torch.randint(0, data.shape[0], (n_points,))
    
    # TODO Note we're sampling all the points here, but we should be sampling a subset
    # Data is small enough that we can take the full set
    # idxs = torch.arange(data.shape[0])
    starts = data[idxs]
    pV = model(starts)
    velocity = Vgpu[idxs]
    # Calculate the loss for each node in the network
    for node in graph.nodes:
        # Get the index of the node in the network data
        node_idx = node_idxs[node]
        # Get the predicted velocity for the node
        node_pV = pV[:,node_idx]
        # Get the true velocity vector for the node
        node_V = velocity[:,node_idx]
        # Compute the loss between the predicted and true velocity vectors
        node_loss = mse(node_pV, node_V)
        # Add the loss to the total loss
        losses[node].append(node_loss.item())
    
    # Compute the loss between the predicted and true velocity vectors
    loss = mse(pV, velocity)
    loss.backward()
    optimizer.step()
    print(i,' '.join([f'{x.item():.9f}' for x in [loss]]), flush=True)
    total_losses.append(loss.item())

    if i%100 == 0:
        plt.plot(total_losses)
        plt.savefig(f'../figures/loss_curve/loss_{genotype}.png')

    # Every N steps plot the predicted and true vectors
    if i % 200 == 0:
        idxs = torch.arange(data.shape[0])
        starts = data[idxs]
        pV = model(starts)
        dpv = embed_velocity(X=tonp(starts),
                             velocity=tonp(pV),
                             embed_fn=embed)
        plot_arrows(idxs=idxs,
                    points=embedding, 
                    V=V_emb*10,
                    pV=dpv*10,
                    sample_every=5,
                    scatter=False,
                    save_file=f'../figures/embedding/vector_field_{genotype}_{i}.png',
                    c=via.single_cell_pt_markov,
                    s=.5,
                    xlimits=x_limits,
                    ylimits=y_limits)
    # if i%1000 == 0:
    #     plot_traces(model, trace_starts, i, n_traces=n_traces)
    #     torch.save(model.state_dict(), 'simple_model.torch')

    plt.close()

#%% 
# Save the trained model
# Get the current datetime to use in the model name
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f'../models/flow_model_{genotype}_{timestamp}.torch')
# %%
