#%%
# %load_ext autoreload
# %autoreload 2
#%%
import torch
import numpy as np
import scanpy as sc
from flow_model import GraphFlowModel
from util import velocity_vectors, is_notebook
from joblib import Parallel, delayed
from datetime import datetime
import os
import sys
from itertools import product
import pickle

#%%
if is_notebook():
    now = datetime.now()
    tmstp = now.strftime("%Y%m%d_%H%M%S")
else:
    if len(sys.argv) < 2:
        print('Usage: python train_graph_flow_model.py <timestamp>')
        sys.exit(1)
    tmstp = sys.argv[1]

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
print(tmstp)
print('Setting up')
adata = sc.read_h5ad(f'../data/{genotype}_{dataset}.h5ad')

#%%
# Get the transition matrix from the VIA graph
X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)

#%%
print('Loading data to GPUs')
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
os.mkdir(f'../output/{tmstp}')
os.mkdir(f'../output/{tmstp}/logs')
os.mkdir(f'../output/{tmstp}/params')
os.mkdir(f'../output/{tmstp}/models')

#%%
print('Initializing model')
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))
nodes = list(adata.uns['id_row'].keys())
#%%
num_nodes = X.shape[1]
hidden_dim = 10
num_layers = 3

device = 'cuda:0'
model = GraphFlowModel(num_layers=num_layers,
                       graph=graph, 
                       data_idxs=adata.uns['id_row'], 
                       hops=3)
for key, node_model in model.models.items():
    torch.nn.init.uniform_(node_model.layers[0].weight, 0, 1)

#%%
# Hyperparameter tuning
gpu_parallel = 5
lmbda_space = [('lmbda',x) for x in np.linspace(1e-3, 1e-2, 10)]
# Prepend a zero to the list of lambdas
lmbda_space = [('lmbda',0.0)] + lmbda_space
# hidden_dim_space = [('hidden_dim',x) for x in np.linspace(10, 100, 10)]
trained_models = {}
outfile = open(f'../output/train_l1_flow_model.out', 'w')
params = list(product(lmbda_space))
pickle.dump(params, open(f'../output/{tmstp}/params/graph_flow_model_{genotype}.pickle', 'wb'))

#%%
def train(node, gpu, param_idx):
    lmbda = dict(params[param_idx])['lmbda']
    logfile = open(f'../output/{tmstp}/logs/graph_flow_model_{node}_{genotype}_{param_idx}.log', 'w')
    if node not in model.models:
        return (node,None)
    model.models[node] = model.models[node].to(f'cuda:{gpu}')
    node_model = model.models[node]
    optimizer = torch.optim.Adam(node_model.parameters(), lr=1e-3)
    node_idx = adata.uns['id_row'][node]

    n_epoch = 10_000
    for epoch in range(n_epoch+1):
        optimizer.zero_grad()
        # Sample a random set of points
        idxs = torch.randint(0, data[gpu].shape[0], (n_points,))
        starts = data[gpu][idxs]
        pV = model(starts, node)
        velocity = Vgpu[gpu][idxs, node_idx][:,None]
        # Compute the loss between the predicted and true velocity vectors
        loss = mse(pV, velocity.squeeze())
        # Compute the L1 penalty on the input weights
        gt0 = node_model.layers[0].weight > 0
        n_active_inputs = ((gt0).sum(axis=0) > 0).sum()
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
        if epoch % 100 == 0:
            msg = f'{node} {epoch} {total_loss:.4e} {loss:.4e} {l1_penalty:.4f} {gt0.sum()} {n_active_inputs}'
            # print(msg)
            logfile.write(msg + '\n')

    return (node, node_model)
# train(key, 0, 0)
#%%
print('Training models')
# Train the individual node models in parallel
for param_idx in range(len(params)):
    print(params, flush=True)
    # verbose>=51 prints the output of joblib to stdout
    # Run num_gpus*gpu_parallel jobs. This assigns gpu_parallel jobs to each gpu
    parallel = Parallel(n_jobs=num_gpus*gpu_parallel, verbose=51)
    trained_models[param_idx] = parallel(delayed(train)(node, i%num_gpus, param_idx) for i,node in list(enumerate(nodes)))
#%%
# Save the dictionary of trained models
state_dicts = {}
for param_idx in trained_models:
    # Remove the None values
    model = {k:v for k,v in trained_models[param_idx] if v is not None}
    state_dicts[param_idx] = model

pickle.dump(state_dicts, open(f'../output/{tmstp}/models/graph_flow_model_{genotype}.pickle', 'wb'))

#%%
# model.models = torch.nn.ModuleList(trained_models)

#%%


