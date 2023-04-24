#%%
%load_ext autoreload
%autoreload 2
#%%
import torch
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from flow_model import L1FlowModel
from util import tonp, plot_arrows, velocity_vectors, embed_velocity, get_plot_limits
from sklearn.decomposition import PCA
# import scvelo as scv

#%%
genotype='wildtype'
dataset = 'net'
#%% 
adata = sc.read_h5ad(f'../data/{genotype}_{dataset}.h5ad')

#%%
pcs = adata.varm['PCs']
pca = PCA()
pca.components_ = pcs.T
pca.mean_ = adata.X.mean(axis=0)

#%%
# Get the transition matrix from the VIA graph
X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)

#%%
def embed(X, pcs=[0,1]):
    return pca.transform(X)[:,pcs]

#%%
embedding = embed(X)
#%%
V_emb = embed_velocity(X, V, embed)

#%%
x_limits, y_limits = get_plot_limits(embedding)
#%%
plot_arrows(idxs=range(len(embedding)), 
            points=np.asarray(embedding), 
            V=V_emb, 
            sample_every=10, 
            c=adata.obs['pseudotime'],
            xlimits=x_limits,
            ylimits=y_limits,
            aw=0.01,)

#%%
num_nodes = X.shape[1]
hidden_dim = 10
num_layers = 3

device = 'cuda:0'
model = L1FlowModel(input_dim=num_nodes, 
                    hidden_dim=hidden_dim, 
                    num_layers=num_layers).to(device)
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')
#%%
data = torch.tensor(X).to(torch.float32).to(device)
Vgpu = torch.tensor(V).to(torch.float32).to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 10000
n_points = 1000
n_traces = 50
n_samples = 10

lmbda = 1

#%%
for i in range(n_epoch+1):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    idxs = torch.randint(0, data.shape[0], (n_points,))
    starts = data[idxs]
    pV = model(starts)
    velocity = Vgpu[idxs]
    # Compute the loss between the predicted and true velocity vectors
    loss = mse(pV, velocity)
    # Compute the L1 penalty on the input weights
    l1_penalty = torch.mean(torch.abs(torch.cat([m.l1 for m in model.module.models])))
    # Add the loss and the penalty
    total_loss = loss + lmbda*l1_penalty
    total_loss.backward()
    optimizer.step()

    # Every N steps plot the predicted and true vectors
    if i % 10 == 0:
        print(i,' '.join([f'{x.item():.9f}' for x in [loss, l1_penalty]]), flush=True)
        
#%%
idxs = torch.arange(data.shape[0])
starts = data[idxs]
pV, _ = model(starts)
dpv = embed_velocity(X=tonp(starts),
                    velocity=tonp(pV),
                    embed_fn=embed)
plot_arrows(idxs=idxs,
            points=embedding, 
            V=V_emb,
            pV=dpv,
            sample_every=5,
            scatter=False,
            save_file=f'../figures/embedding/connected_vector_field_{genotype}_{i}.png',
            c=adata.obs['pseudotime'],
            s=.5,
            xlimits=x_limits,
            ylimits=y_limits)

#%% 
# Save the trained model
# Get the current datetime to use in the model name
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f'../models/l1_flow_model_{genotype}_{timestamp}.torch')
# %%
