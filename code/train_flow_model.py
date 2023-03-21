#%%
import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
from pyVIA.core import l2_norm

#%%
genotype='mutant'

#%% 
# Load the VIA object from pickle
via = pickle.load(open(f'../data/{genotype}_pseudotime.pickle', 'rb'))

#%%
# Load the umap embedding
embedding = pickle.load(open(f'../data/umap_embedding_{genotype}.pickle', 'rb'))
# Load the umap object
umap_ = pickle.load(open(f'../data/umap_{genotype}.pickle', 'rb'))
# Load the pca object
pca_ = pickle.load(open(f'../data/pca_{genotype}.pickle', 'rb'))

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
def plot_arrows(points, V, pV=None, sample_factor=10, save_file=None):
    # Plot the vectors from the sampled points to the transition points
    # Increase the size of the plot to see the vectors
    plt.figure(figsize=(15,15))
    plt.scatter(points[:,0], points[:,1], s=3)
    # black = true vectors
    # Green = predicted vectors
    for i in range(0, len(V), sample_factor):
        plt.arrow(points[i,0], points[i,1], V[i,0], V[i,1], color='black', alpha=1, width=0.015)
        if pV is not None:
            plt.arrow(points[i,0], points[i,1], pV[i,0], pV[i,1], color='g', alpha=1, width=0.015)

    if save_file is not None:
        plt.savefig(save_file)

V_emb = via._velocity_embedding(embedding, smooth_transition=1, b=10)
#%%
plot_arrows(embedding, V_emb, sample_factor=2)

#%%
def plot_traces(model, trace_starts, i, n_traces=50):
    # Generate some sample traces
    traces = model.trajectory(state=trace_starts, tspan=torch.linspace(0, 100, 500))
    trace_plot = traces.cpu().detach().numpy()
    # Create a new figure
    fig, ax = plt.subplots()
    # Plot the data with partial transparency so that we can highlight the traces
    ax.scatter(data[:,0], data[:,1], s=.25, alpha=0.1)
    print(trace_plot.shape)
    for trace in range(n_traces):
        print(trace, n_traces)
        # Plot the traces
        ax.scatter(trace_plot[:,trace,0], trace_plot[:,trace,1], s=1)
    # Save the plot to a file indicating the epoch
    plt.savefig(f'figures/test/traces_{i}.png')

#%%
device = 'cuda:0'
model = WholeCell(input_dim=num_nodes, 
                  output_dim=num_nodes, 
                  hidden_dim=12, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')
sampled = sampled.to(device)
V = torch.tensor(V).to(torch.float32).to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 10000
n_points = 1000
n_traces = 50
trace_starts = sampled[torch.randint(0, sampled.shape[0], (n_traces,))]
n_samples = 10

#%%
for i in range(n_epoch):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    # idxs = torch.randint(0, sampled.shape[0], (n_points,))
    
    # TODO Note we're sampling all the points here, but we should be sampling a subset
    # Data is small enough that we can take the full set
    idxs = torch.arange(sampled.shape[0])
    starts = sampled[idxs]
    # TODO do we need to change the tspan?
    # tspan is a single step from zero to one
    _, fx = model(starts, tspan=torch.linspace(0,1,2))
    velocity = V[idxs]
    # Compute the loss between the predicted and true velocity vectors
    loss = mse(fx[-1], velocity)
    loss.backward()
    optimizer.step()
    print(i,' '.join([f'{x.item():.9f}' for x in 
          [loss]]), flush=True)
    if i % 100 == 0:
        plot_arrows(sampled.detach().cpu(), 
                    fx[-1].detach().cpu(),
                    velocity.detach().cpu(),
                    sample_factor=10,
                    save_file=f'figures/test/vector_field_{i}.png')
    if i%1000 == 0:
        plot_traces(model, trace_starts, i, n_traces=n_traces)
        torch.save(model.state_dict(), 'simple_model.torch')

    plt.close()
# %%
