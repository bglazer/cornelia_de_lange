#%%
import torch
import scanpy as sc
import numpy as np
from flow_model import ConnectedFlowModel
from util import tonp, plot_arrows, velocity_vectors, embed_velocity, is_notebook, get_plot_limits
import sys
from datetime import datetime
import os
from sklearn.decomposition import PCA

#%%
genotype='wildtype'
dataset = 'net'

#%%
if is_notebook():
    now = datetime.now()
    tmstp = now.strftime("%Y%m%d_%H%M%S")
    logfile = sys.stdout
else:
    if len(sys.argv) < 2:
        print('Usage: python train_fully_connected_flow_model.py <timestamp>')
        sys.exit(1)
    tmstp = sys.argv[1]

    os.mkdir(f'../output/{tmstp}')
    os.mkdir(f'../output/{tmstp}/logs')
    os.mkdir(f'../output/{tmstp}/models')
    logfile = open(f'../output/{tmstp}/logs/connected_model_{genotype}.log', 'w')

print(tmstp)
print('Setting up')
adata = sc.read_h5ad(f'../data/{genotype}_{dataset}.h5ad')
device = 'cuda:0'

#%%
X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)

#%%
pct_train = 0.8
n_train = int(pct_train*X.shape[0])
n_val = X.shape[0] - n_train
train_idxs = np.random.choice(X.shape[0], n_train, replace=False)
val_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)
X_train = X[train_idxs]
X_val = X[val_idxs]
V_train = V[train_idxs]
V_val = V[val_idxs]

train_data = torch.tensor(X_train).to(torch.float32).to(device)
val_data = torch.tensor(X_val).to(torch.float32).to(device)
train_V = torch.tensor(V_train).to(torch.float32).to(device)
val_V = torch.tensor(V_val).to(torch.float32).to(device)

#%%
num_nodes = X.shape[1]
hidden_dim = num_nodes*2
num_layers = 3

model = ConnectedFlowModel(input_dim=num_nodes, 
                            output_dim=num_nodes, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')
full_data = torch.tensor(X).to(torch.float32).to(device)
full_V = torch.tensor(V).to(torch.float32).to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 10_000
n_points = 1_000

#%%
for i in range(n_epoch+1):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    idxs = torch.randint(0, train_data.shape[0], (n_points,))
    
    # TODO Note we're sampling all the points here, but we should be sampling a subset
    # Data is small enough that we can take the full set
    # idxs = torch.arange(data.shape[0])
    starts = train_data[idxs]
    # TODO do we need to change the tspan?
    # tspan is a single step from zero to one
    pV = model(starts)
    velocity = train_V[idxs]
    # Compute the loss between the predicted and true velocity vectors
    loss = mse(pV, velocity)
    loss.backward()
    optimizer.step()
    logfile.write(f'{i} Train loss: {loss.item():.9f}\n')
    if i%100 == 0:
        # Run the model on the validation set
        pV_val = model(val_data)
        val_loss = mse(pV_val, val_V)
        logfile.write(f'{i} Validation loss: {val_loss.item():.9f}\n')
#%%
if not is_notebook():
    logfile.close()
    torch.save(model.state_dict(), 
            f'../output/{tmstp}/models/connected_model_{genotype}.torch')
# %%
pcs = adata.varm['PCs']
pca = PCA()
pca.components_ = pcs.T
pca.mean_ = adata.X.mean(axis=0)

#%%
def embed(X, pcs=[0,1]):
    return pca.transform(X)[:,pcs]

#%%
# Combine train and validation data
embedding = np.array(embed(X))
V_emb = np.array(embed_velocity(X, V, embed))
pV = model(full_data)
dpv = embed_velocity(X=tonp(full_data),
                    velocity=tonp(pV),
                    embed_fn=embed,)
idxs = np.arange(0, embedding.shape[0], 1)
x_limits, y_limits = get_plot_limits(embedding)
plot_arrows(idxs=idxs,
            points=embedding, 
            V=V_emb,
            pV=dpv,
            sample_every=5,
            scatter=False,
            save_file=f'../figures/embedding/vector_field_wildtype_fully_connected_{tmstp}.png',
            c=adata.obs['pseudotime'],
            s=1.5,
            xlimits=x_limits,
            ylimits=y_limits)
            
# %%
