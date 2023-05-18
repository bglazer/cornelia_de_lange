#%%
# %load_ext autoreload
# %autoreload 2
#%%
import torch
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from flow_model import L1FlowModel
from util import tonp, plot_arrows, velocity_vectors, embed_velocity, get_plot_limits, is_notebook
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from datetime import datetime
import os
import sys
import pickle
from itertools import product

#%%
genotype='wildtype'
dataset = 'net'

#%% 
# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

#%%
if is_notebook():
    now = datetime.now()
    tmstp = now.strftime("%Y%m%d_%H%M%S")
else:
    if len(sys.argv) < 2:
        print('Usage: python train_l1_flow_model.py <timestamp>')
        sys.exit(1)
    tmstp = sys.argv[1]

    os.mkdir(f'../output/{tmstp}')
    os.mkdir(f'../output/{tmstp}/logs')
    os.mkdir(f'../output/{tmstp}/models')


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
pct_train = 0.8
n_train = int(pct_train*X.shape[0])
n_val = X.shape[0] - n_train
train_idxs = np.random.choice(X.shape[0], n_train, replace=False)
val_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)
X_train = X[train_idxs]
X_val = X[val_idxs]
V_train = V[train_idxs]
V_val = V[val_idxs]

device='cuda:1'
train_data = torch.tensor(X_train).to(torch.float32).to(device)
val_data = torch.tensor(X_val).to(torch.float32).to(device)
train_V = torch.tensor(V_train).to(torch.float32).to(device)
val_V = torch.tensor(V_val).to(torch.float32).to(device)

#%%
embedding = embed(X)
#%%
V_emb = embed_velocity(X, V, embed)

#%%
x_limits, y_limits = get_plot_limits(embedding)
#%%
# plot_arrows(idxs=range(len(embedding)), 
#             points=np.asarray(embedding), 
#             V=V_emb, 
#             sample_every=10, 
#             c=adata.obs['pseudotime'],
#             xlimits=x_limits,
#             ylimits=y_limits,
#             aw=0.01,)

#%%
num_gpus = 4
data = [torch.tensor(X).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
Vgpu = [torch.tensor(V).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_points = 1000
n_traces = 50
n_samples = 10

loss_fn = torch.nn.MSELoss(reduction='mean')

#%%
n_epoch = 10_000
num_nodes = X.shape[1]
hidden_dim = 10
num_layers = 3

device = 'cuda:0'
model = L1FlowModel(input_dim=num_nodes, 
                    hidden_dim=hidden_dim, 
                    num_layers=num_layers)
for node_model in model.models:
    torch.nn.init.uniform_(node_model.layers[0].weight, 0, 1)

#%%
def train(model_idx, gpu, params, idx):
    lmbda = params['lmbda']
    if is_notebook():
        logfile = sys.stdout
    else:
        logfile = open(f'../output/{tmstp}/logs/l1_flow_model_{model_idx}_{genotype}_{idx}.log', 'w')
    node_model = model.models[model_idx].to(f'cuda:{gpu}')
    optimizer = torch.optim.Adam(node_model.parameters(), lr=1e-3)

    for epoch in range(n_epoch+1):
        optimizer.zero_grad()
        # Run the model from N randomly selected data points
        # Random sampling
        idxs = torch.randint(0, data[gpu].shape[0], (n_points,))
        starts = data[gpu][idxs]
        pV = node_model(starts)
        velocity = Vgpu[gpu][idxs, model_idx][:,None]
        # Compute the loss between the predicted and true velocity vectors
        loss = mse(pV, velocity)
        # Compute the L1 penalty on the input weights
        gt0 = node_model.layers[0].weight > 0
        n_active_inputs = ((model.models[0].layers[0].weight > 0).sum(axis=0) > 0).sum()
        if gt0.sum() > 0:
            l1_penalty = torch.mean(torch.abs(node_model.layers[0].weight[gt0]))
            # Add the loss and the penalty
            total_loss = loss + lmbda*l1_penalty
        else:
            total_loss = loss
            l1_penalty = 0
        total_loss.backward()
        optimizer.step()

        # Every N steps plot the predicted and true vectors
        if epoch % 10 == 0:
            msg = f'{model_idx} {epoch} {total_loss:.4e} {loss:.4e} {l1_penalty:.4f} {gt0.sum():.4f} {n_active_inputs}'
            # print(msg)
            logfile.write(msg + '\n')

    return node_model

#%%
# Hyperparameter tuning
gpu_parallel = 5
lmbda_space = [('lmbda',x) for x in np.linspace(1e-3, 1e-2, 10)]
lmbda_space = [('lmbda',0)] + lmbda_space
# If there are more params, 
params = [dict(x) for x in zip(lmbda_space)]
#%%
train(0,0,params[1],0)
#%%
# hidden_dim_space = [('hidden_dim',x) for x in np.linspace(10, 100, 10)]
trained_models = {}
outfile = open(f'../output/train_l1_flow_model.out', 'w')
params = list(product(lmbda_space))
pickle.dump(params, open(f'../output/{tmstp}/params/l1_flow_model_{genotype}.pickle', 'wb'))
for param_idx in range(len(params)):
    print(params, flush=True)
    parallel = Parallel(n_jobs=num_gpus*gpu_parallel)
    trained_models[param_idx] = parallel(delayed(train)(i, i%num_gpus, params[param_idx]) for i in range(num_nodes))
#%%
# Save the dictionary of trained models
state_dicts = {}
for model in trained_models:
    state_dicts[model] = torch.nn.ModuleList(trained_models[model]).state_dict()

pickle.dump(state_dicts, open(f'../output/{tmstp}/models/l1_flow_model_{genotype}.pickle', 'wb'))

#%%
# model.models = torch.nn.ModuleList(trained_models)

#%%
# gpu = 0
# idxs = torch.arange(data[gpu].shape[0])
# starts = data[gpu][idxs]
# model = model.to(f'cuda:{gpu}')
# pV = model(starts)
# dpv = embed_velocity(X=tonp(starts),
#                     velocity=tonp(pV),
#                     embed_fn=embed)
# plot_arrows(idxs=idxs,
#             points=embedding, 
#             V=V_emb,
#             pV=dpv,
#             sample_every=5,
#             scatter=False,
#             save_file=f'../figures/embedding/l1_vector_field_{genotype}_{tmstp}.png',
#             c=adata.obs['pseudotime'],
#             s=1.5,
#             xlimits=x_limits,
#             ylimits=y_limits)

