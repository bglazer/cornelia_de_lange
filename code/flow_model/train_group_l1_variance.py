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
import json
from copy import deepcopy
from datetime import datetime, timedelta

#%%
start_tmstp = datetime.now().timestamp()

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
V_vars = np.zeros_like(V)
for i in range(V.shape[0]):
    V_vars[i] = np.var(X[T[i].indices], axis=0)
V_vars[np.isnan(V_vars)] = 0.0
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
V_var_train = V_vars[train_idxs]
V_var_val = V_vars[val_idxs]

num_gpus = 4
train_data = [torch.tensor(X_train).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
val_data = [torch.tensor(X_val).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
train_V = [torch.tensor(V_train).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
val_V = [torch.tensor(V_val).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
train_V_var = [torch.tensor(V_var_train).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]
val_V_var = [torch.tensor(V_var_val).to(torch.float32).to(f'cuda:{i}') for i in range(num_gpus)]

#%%
mse = torch.nn.MSELoss(reduction='mean')

#%%
n_points = 1000
n_epoch = 5_000
# Threshold for stopping training if the validation loss does not change by more than this amount
delta_loss_threshold = 5e-3

# Average over the last N validation losses
loss_accumulation = 5
num_nodes = X.shape[1]
hidden_dim = 32
num_layers = 3
checkpoint_interval = 100

rewind_checkpoints = {}

model = GroupL1FlowModel(input_dim=num_nodes, 
                         hidden_dim=hidden_dim, 
                         num_layers=num_layers,
                         predict_var=True)

#%%
def train(model_idx, gpu, l1_alphas):
    if is_notebook():
        logfile = sys.stdout
    else:
        logfile = open(f'{outdir}/logs/l1_flow_model_{model_idx}_{genotype}.log', 'w')
    node_model = model.models[model_idx].to(f'cuda:{gpu}')
    optimizer = torch.optim.Adam(node_model.parameters(), lr=5e-4)

    models = []

    best_model = None
    loss_queue = deque(maxlen=loss_accumulation)
    for iteration, l1_alpha in enumerate(l1_alphas):
        for epoch in range(n_epoch+1):
            optimizer.zero_grad()
            # Run the model from N randomly selected data points
            # Random sampling
            idxs = torch.randint(0, train_data[gpu].shape[0], (n_points,))
            starts = train_data[gpu][idxs]
            y = node_model(starts)
            pV = y[:,0,None]
            pVar = y[:,1,None]
            velocity = train_V[gpu][idxs, model_idx][:,None]
            variance = train_V_var[gpu][idxs, model_idx][:,None]
            # Compute the loss between the predicted and true velocity vectors
            mean_loss = mse(pV, velocity)
            var_loss = mse(pVar, variance)
            l1 = node_model.group_l1.mean()
            loss = mean_loss + var_loss + l1*l1_alpha
            # Compute the L1 penalty on the input weights
            loss.backward()
            optimizer.step()

            # Every N steps calculate the validation loss
            if epoch % checkpoint_interval == 0:
                if iteration == 0:
                    rewind_checkpoints[epoch] = deepcopy(node_model.state_dict())
                
                # Run the model on the validation set
                val_y = node_model(val_data[gpu])
                val_pV = val_y[:,0,None]
                val_pVar = val_y[:,1,None]
                val_mean_loss = mse(val_pV, val_V[gpu][:,model_idx][:,None])
                val_var_loss = mse(val_pVar, val_V_var[gpu][:,model_idx][:,None])
                val_loss = val_mean_loss + val_var_loss + l1*l1_alpha
                
                loss_queue.append(val_loss.item())
                pct_var = np.std(loss_queue)/np.mean(loss_queue)
        
                active_inputs = int((node_model.group_l1 > 0).sum())

                logline = {
                    'epoch':           epoch,
                    'train_loss':      loss.item(),
                    'train_var_loss':  var_loss.item(),
                    'val_mean_loss':   val_mean_loss.item(),
                    'val_var_loss':    val_var_loss.item(),
                    'l1_loss':         l1.item(),
                    'active_inputs':   active_inputs,
                    'pct_var':         pct_var,
                    'l1_alpha':        l1_alpha,
                }
                logfile.write(json.dumps(logline) + '\n')
                # We have enough losses in the queue to evaluate the stopping criteria
                accumulated_losses = len(loss_queue) == loss_accumulation
                # If its the first round or we've pruned some inputs
                pruned_or_first = (iteration == 0 or active_inputs < num_nodes)
                # If the validation loss has not changed by more than the threshold
                no_change = abs(pct_var) < delta_loss_threshold
                conditions = [accumulated_losses, pruned_or_first, no_change]
                if all(conditions):
                    break
        logline = {
            'iteration': iteration,
            'l1_alpha': l1_alpha,
            'val_mean_loss':   val_mean_loss.item(),
            'val_var_loss':    val_var_loss.item(),
            'active_inputs': active_inputs,
        }
        logfile.write(json.dumps(logline) + '\n')
        if iteration == 0:
            rewind_epoch = (int(epoch * .05) // checkpoint_interval) * checkpoint_interval
            rewind_epoch = max(rewind_epoch, checkpoint_interval)
            rewind_checkpoint = rewind_checkpoints[rewind_epoch]

        models.append(deepcopy(node_model.state_dict()))
        node_model.load_state_dict(rewind_checkpoint)


    return models

# %%
# model_idx = 0
# %%
# model_idx += 1
# nms = train(model_idx, 0, list(np.linspace(.01, .6, 6)) + [100.0])

#%%
gpu_parallel = 5

trained_models = {}
# Start with zero penalty, then increase, then finish with very high penalty to induce total pruning
l1_alphas = [.6] #[0.0] + list(np.linspace(.01, .6, 6)) + [100.0]
note = '''Trained with group L1 penalty, pruned during training.'''
params = {
    'l1_alphas': l1_alphas,
    'prune_type': 'Group L1 penalty, pruned during training',
    'weight_penalty': 'l1',
    'note': note,
}
pickle.dump(params, open(f'{outdir}/params/group_l1_variance_model_{genotype}.pickle', 'wb'))
parallel = Parallel(n_jobs=num_gpus*gpu_parallel)
trained_models = parallel(delayed(train)(i, i%num_gpus, params['l1_alphas'])
                          for i in range(num_nodes))

model_filename = f'{outdir}/models/group_l1_variance_model_{genotype}.pickle'
with open(model_filename, 'wb') as model_file:
    pickle.dump(trained_models, model_file)

end_tmstp = datetime.now().timestamp()
tmstp_diff = end_tmstp - start_tmstp
# Format the time difference as number of hours, minutes, and seconds
tmstp_diff = str(timedelta(seconds=tmstp_diff))
print(f'Training time: {tmstp_diff}')