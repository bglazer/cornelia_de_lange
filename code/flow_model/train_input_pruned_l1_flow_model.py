#%%
# %load_ext autoreload
# %autoreload 2
#%%
import torch
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
from flow_model import L1FlowModel
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import os
import sys
import pickle
from collections import deque
# Import torch pruning 
from torch.nn.utils import prune
import json
from math import ceil
#%%
# util is in the parent directory, so we need to add it to the path
sys.path.append('..')
from util import tonp, velocity_vectors, sliding_window
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
if len(sys.argv) < 2:
    print('Usage: python train_l1_flow_model.py <timestamp>')
    sys.exit(1)
tmstp = sys.argv[1]
outdir = f'../../output/{tmstp}'
os.mkdir(f'{outdir}')
os.mkdir(f'{outdir}/logs')
os.mkdir(f'{outdir}/models')
os.mkdir(f'{outdir}/params')


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
pct_train = 0.8
n_train = int(pct_train*X.shape[0])
n_val = X.shape[0] - n_train
train_idxs = np.random.choice(X.shape[0], n_train, replace=False)
val_idxs = np.setdiff1d(np.arange(X.shape[0]), train_idxs)
X_train = X[train_idxs]
X_val = X[val_idxs]
V_train = V[train_idxs]
V_val = V[val_idxs]

num_gpus = 4
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
n_epoch = 5_000
# Threshold for stopping training if the validation loss does not change by more than this amount
delta_loss_threshold = 1e-2
# Threshold for stopping training if the validation loss increases by more than this amount
increased_loss_threshold = 1e-1

# Average over the last N validation losses
diff_accumulation = 5
num_nodes = X.shape[1]
hidden_dim = 32
num_layers = 3

model = L1FlowModel(input_dim=num_nodes, 
                    hidden_dim=hidden_dim, 
                    num_layers=num_layers)

#%%
def train(model_idx, gpu, prune_amt):
    logfile = open(f'{outdir}/logs/l1_flow_model_{model_idx}_{genotype}.log', 'w')
    # logfile = sys.stdout
    node_model = model.models[model_idx].to(f'cuda:{gpu}')
    
    optimizer = torch.optim.Adam(node_model.parameters(), lr=1e-3)

    models = []

    linear_layers = [i for i,layer in enumerate(node_model.layers[:1])
                     if type(layer) is torch.nn.Linear]
    layer_sizes = np.array([node_model.layers[i].weight.shape[1]
                            for i in linear_layers])
    num_nodes = int(layer_sizes.sum())
    layer_pcts = layer_sizes/num_nodes

    if type(prune_amt) == int:
        # Calculat an evenly spaced series of pruning steps, 
        # np.linspace correctly handles calculating the last step size 
        # that will get to 1 node
        n_rounds = num_nodes//prune_amt
        steps = np.linspace(num_nodes, 0, n_rounds, dtype=int)
        prune_steps = [a-b for a,b in sliding_window(steps, 2)] + [0]
    elif type(prune_amt) == float:
        # Calculate the number of percentage pruning rounds required to get to a single input
        n_rounds = int(np.log(1.0/num_nodes)/np.log(1-prune_amt))
        prune_steps = [prune_amt]*n_rounds
    else:
        raise ValueError('prune_amt must be an int or a float')
    
    n_rounds = len(prune_steps)

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
            l1 = torch.tensor([node_model.layers[i].weight.abs().mean() 
                               for i in linear_layers]).mean()
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
                
                if epoch > 0:
                    diff = val_loss.item() - best_val_loss
                    diffs.append(diff)
                    avg_diff = np.mean(diffs)/best_val_loss
                
                logline = {'epoch': epoch, 
                           'round': round,
                           'train_loss': loss.item(),
                           'validation_loss': val_loss.item()}
                logfile.write(json.dumps(logline) + '\n')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model = node_model.state_dict()
                    diffs.clear()
        
                if len(diffs) == diff_accumulation:
                    # Early stop if the validation loss is getting worse
                    if avg_diff > increased_loss_threshold:
                        break
                    # Early stop if the validation loss is not changing
                    if abs(avg_diff) < delta_loss_threshold:
                        break
        
        models.append(best_model)
        node_model.load_state_dict(best_model)
        if round > 0:
            active_nodes = [int((node_model.layers[i].weight_mask.sum(dim=0) > 0).sum()) 
                            for i in linear_layers]
        else:
            active_nodes = [int(node_model.layers[i].weight.shape[1]) 
                            for i in linear_layers]
        # L1 prune the input weights of the model
        logline = {'round': round,
                   'best_val_loss': best_val_loss,
                   'active_nodes': active_nodes}
        logfile.write(json.dumps(logline) + '\n')

        prune_step = prune_steps[round]
            
        for i, layer_idx in enumerate(linear_layers):
            layer_pct = layer_pcts[i]
            step = ceil(prune_step*layer_pct)
            if step > active_nodes[i]:
                step = active_nodes[i]
            node_model.layers[layer_idx] = prune.ln_structured(module=node_model.layers[layer_idx], 
                                                               name='weight', 
                                                               amount=step,
                                                               n=1,
                                                               dim=1)

    return models

#%%
model_idx = 51
nm=train(model_idx, 0, 50)
#%%
model.models[model_idx].load_state_dict(nm[3])

#%%
gpu_parallel = 5

trained_models = {}
params = {
    'prune_amt': 10,
    'prune_type': 'l1_structured_input',
    'weight_penalty': 'l1'
}
pickle.dump(params, open(f'{outdir}/params/pruned_l1_flow_model_{genotype}.pickle', 'wb'))
parallel = Parallel(n_jobs=num_gpus*gpu_parallel)
trained_models = parallel(delayed(train)(i, i%num_gpus, params['prune_amt']) 
                          for i in range(num_nodes))

pickle.dump(trained_models, open(f'{outdir}/models/pruned_l1_flow_model_{genotype}.pickle', 'wb'))
