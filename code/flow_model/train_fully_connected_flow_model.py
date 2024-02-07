#%%
import torch
import scanpy as sc
import numpy as np
from flow_model import ConnectedFlowModel
import sys
import os
import json
sys.path.append('..')
from util import velocity_vectors

#%%
genotype='wildtype'
dataset = 'net'

#%%
if len(sys.argv) < 2:
    print('Usage: python train_fully_connected_flow_model.py <timestamp>')
    sys.exit(1)
# tmstp = 0
tmstp = sys.argv[1]

outdir = f'../../output/{tmstp}'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    os.mkdir(f'{outdir}/logs')
    os.mkdir(f'{outdir}/models')
logfile = open(f'{outdir}/logs/connected_model_{genotype}.log', 'w')
#%%
print(tmstp)
print('Setting up')
adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')

#%%
X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)
V_vars = np.zeros_like(V)
for i in range(V.shape[0]):
    V_vars[i] = np.var(X[T[i].indices], axis=0)
V_vars[np.isnan(V_vars)] = 0.0

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

device = 'cuda:0'
num_gpus = train_data = torch.tensor(X_train).to(torch.float32).to(device)
val_data = torch.tensor(X_val).to(torch.float32).to(device)
train_V = torch.tensor(V_train).to(torch.float32).to(device)
val_V = torch.tensor(V_val).to(torch.float32).to(device)
train_V_var = torch.tensor(V_var_train).to(torch.float32).to(device)
val_V_var = torch.tensor(V_var_val).to(torch.float32).to(device)

#%%
num_nodes = X.shape[1]
hidden_dim = num_nodes*2
num_layers = 3

model = ConnectedFlowModel(input_dim=num_nodes, 
                            output_dim=num_nodes, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            predict_var=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')
full_data = torch.tensor(X).to(torch.float32).to(device)
full_V = torch.tensor(V).to(torch.float32).to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 5_000
n_points = 1_000

best_validation_loss = float('inf')
#%%
for i in range(n_epoch+1):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    idxs = torch.randint(0, train_data.shape[0], (n_points,))
    
    starts = train_data[idxs]
    pV, pVar = model(starts)
    velocity = train_V[idxs]
    variance = train_V_var[idxs]
    # Compute the loss between the predicted and true velocity vectors
    mean_loss = mse(pV, velocity)
    var_loss = mse(pVar, variance)
    loss = mean_loss + var_loss
    # Compute the loss between the predicted and true velocity vectors
    loss = mse(pV, velocity)
    loss.backward()
    optimizer.step()
    logline = {
        'epoch': i,
        'train_loss': loss.item(),
    }
    logfile.write(json.dumps(logline) + '\n')
    if i%10 == 0:
        # Run the model on the validation set
        pV_val, pVar_val = model(val_data)
        val_loss = mse(pV_val, val_V)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
        logline = {
            'epoch': i,
            'mean_validation_loss': val_loss.item(),
            'var_validation_loss': mse(pVar_val, val_V_var).item(),
        }
        logfile.write(json.dumps(logline) + '\n')

print('Best validation loss:', best_validation_loss.item())

logfile.close()
torch.save(model.state_dict(), 
        f'{outdir}/models/connected_model_{genotype}.torch')

# %%
