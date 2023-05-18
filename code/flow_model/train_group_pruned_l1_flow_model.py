#%%
# %load_ext autoreload
# %autoreload 2
#%%
import torch
import numpy as np
import scanpy as sc
from flow_model import GroupL1FlowModel
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from datetime import datetime
import os
import sys
import pickle
from collections import deque
from torch.nn.utils import prune
#%%
# util is in the parent directory, so we need to add it to the path
sys.path.append('..')
from util import velocity_vectors, is_notebook
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
    outdir = f'../../output/{tmstp}'
    os.mkdir(f'{outdir}')
    os.mkdir(f'{outdir}/logs')
    os.mkdir(f'{outdir}/models')


#%% 
adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')

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
    return np.array(pca.transform(X)[:,pcs])

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

# TODO change to 4 GPUs
num_gpus = 1
train_data = [torch.tensor(X_train).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
val_data = [torch.tensor(X_val).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
train_V = [torch.tensor(V_train).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
val_V = [torch.tensor(V_val).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]

#%%
mse = torch.nn.MSELoss(reduction='mean')

n_points = 1000
n_traces = 50
n_samples = 10

#%%
n_epoch = 10_000
# Threshold for stopping training if the validation loss does not change by more than this amount
delta_loss_threshold = 1e-2
# Threshold for stopping training if the validation loss increases by more than this amount
increased_loss_threshold = 1e-1

# Average over the last N validation losses
diff_accumulation = 5
num_nodes = X.shape[1]
hidden_dim = 32
num_layers = 3

model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers)

#%%
def train(model_idx, gpu, prune_pct, n_rounds):
    if is_notebook():
        logfile = sys.stdout
    else:
        logfile = open(f'{outdir}/logs/l1_flow_model_{model_idx}_{genotype}.log', 'w')
    node_model = model.models[model_idx].to(f'cuda:{gpu}')
    optimizer = torch.optim.Adam(node_model.parameters(), lr=1e-3)

    models = []

    for round in range(n_rounds):
        best_model = None
        diffs = deque(maxlen=diff_accumulation)
        best_val_loss = np.inf
        for epoch in range(n_epoch+1):
            optimizer.zero_grad()
            # Run the model from N randomly selected data points
            # Random sampling
            idxs = torch.randint(0, train_data[gpu].shape[0], (n_points,))
            starts = train_data[gpu][idxs]
            pV = node_model(starts)
            velocity = train_V[gpu][idxs, model_idx][:,None]
            # Compute the loss between the predicted and true velocity vectors
            loss = mse(pV, velocity)
            l1 = torch.abs(node_model.group_l1).sum()
            total = loss + l1
            # Compute the L1 penalty on the input weights
            total.backward()
            optimizer.step()

            # Every N steps calculate the validation loss
            if epoch % 100 == 0:
                # Run the model on the validation set
                val_pV = node_model(val_data[gpu])
                val_loss = mse(val_pV, val_V[gpu][:,model_idx][:,None])
                # Save the model if it has the lowest validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model = node_model
                
                diff = val_loss.item() - best_val_loss
                diffs.append(diff)
                avg_diff = np.mean(diffs)/best_val_loss
        
                logfile.write(f'validation_diff: {model_idx:4d} {epoch:8d} {avg_diff:.4f}\n')
                logfile.write(f'train_loss:      {model_idx:4d} {epoch:8d} {loss:.4e}\n')
                logfile.write(f'validation_loss: {model_idx:4d} {epoch:8d} {val_loss.item():.4e}\n')
                if len(diffs) == diff_accumulation:
                    # Early stop if the validation loss is getting worse
                    if avg_diff > increased_loss_threshold:
                        break
                    # Early stop if the validation loss is not changing
                    if abs(avg_diff) < delta_loss_threshold:
                        break
        print(f'**best_val_loss: {best_val_loss:.4e}')
        node_model = best_model
        # L1 prune the input weights of the model
        node_model = prune.l1_unstructured(node_model, 
                                           name='group_l1', 
                                           amount=prune_pct)
        models.append(node_model.state_dict())
        logfile.write(f'active_inputs: {int(node_model.group_l1_mask.sum()):d}\n')

    return models

#%%
train(0, 0, .2, 5)

#%%
# Hyperparameter tuning
gpu_parallel = 5

# hidden_dim_space = [('hidden_dim',x) for x in np.linspace(10, 100, 10)]
trained_models = {}
# outfile = open(f'{outdir}/train_pruned_l1_flow_model.out', 'w')
params = list()
# pickle.dump(params, open(f'{outdir}/{tmstp}/params/pruned_l1_flow_model_{genotype}.pickle', 'wb'))
# for param_idx in range(len(params)):
#     print(params, flush=True)
parallel = Parallel(n_jobs=num_gpus*gpu_parallel)
trained_models = parallel(delayed(train)(i, i%num_gpus, .2, 20) for i in range(num_nodes))
#%%
# Save the dictionary of trained models
state_dicts = {}
for model in trained_models:
    state_dicts[model] = torch.nn.ModuleList(trained_models[model]).state_dict()

pickle.dump(state_dicts, open(f'{outdir}/models/pruned_l1_flow_model_{genotype}.pickle', 'wb'))
