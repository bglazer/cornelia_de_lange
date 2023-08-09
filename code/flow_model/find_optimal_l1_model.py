#%%
import pickle
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
from flow_model import GroupL1FlowModel
import sys
sys.path.append('..')
from util import embed_velocity, velocity_vectors
import scanpy as sc
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
#%%
genotype = 'mutant'
tmstp = '20230608_093734'
adata = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')

#%%
losses = {}
active_inputs = {}

logfiles = glob.glob(f'../../output/{tmstp}/logs/*.log')
for path in logfiles:
    filename = path.split('/')[-1]
    node = int(filename.split('_')[3])
    with open(path, 'r') as f:
        for line in f:
            log = json.loads(line)
            if 'epoch' not in log:
                if node not in losses:
                    losses[node] = []
                    active_inputs[node] = []
                losses[node].append(log['val_mean_loss'])
                active_inputs[node].append(log['active_inputs'])
        
# %%
nodes = list(losses.keys())
best_idxs = {}
no_active = []

for node in tqdm(nodes):
    # print(node)
    losses[node] = np.asarray(losses[node])
    active_inputs[node] = np.asarray(active_inputs[node])
    loss = losses[node]
    active = active_inputs[node]

    max_loss = np.max(loss)
    min_loss = np.min(loss)
    max_active = np.max(active)
    min_active = np.min(active)
    # If there aren't any values with active nodes, just say min_idx is 0
    if max_active == 0:
        best_idxs[node] = 0
        no_active.append(node)
        continue
    # If all values have the same number of active nodes, min_idx is the best loss
    if max_active == min_active:
        best_idxs[node] = np.argmin(loss)
        continue
    if max_loss == min_loss:
        best_idxs[node] = np.argmin(active)
        continue
    # Normalize losses and active nodes
    nrm_loss = (loss - min_loss) / (max_loss - min_loss)
    nrm_active = (active - min_active) / (max_active - min_active)
    # Remove indexes that have zero active nodes
    one_active = np.where(active > 0)[0]
    nrm_loss = nrm_loss[one_active]
    nrm_active = nrm_active[one_active]
    # Find the value that minimizes loss and number of active nodes
    # i.e. min vector norm of the two normalized values
    norm = np.sqrt(nrm_loss**2 + nrm_active**2)
    # Find the index of the minimum norm
    min_idx = np.argmin(norm)
    best_idxs[node] = one_active[min_idx]
    # Plot the validation loss versus active nodes for each model
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    axs.scatter(loss[one_active], active[one_active], s=4, c=norm, cmap='viridis')
    axs.scatter(loss[best_idxs[node]], active[best_idxs[node]], s=8, c='r')
    axs.set_xlabel('Validation loss')
    axs.set_ylabel('Number of active inputs')
    gene = adata.var_names[node]
    axs.set_title(f'{gene}')
    plt.tight_layout()
    plt.savefig(f'../../figures/loss_vs_active_nodes/{node}_{gene}_{genotype}_{tmstp}.png')
    plt.close()
#%%
minimal_idxs = {}
min_num_actives =  []
for node in tqdm(nodes):
    # print(node)
    losses[node] = np.asarray(losses[node])
    active_inputs[node] = np.asarray(active_inputs[node])
    loss = losses[node]
    active = active_inputs[node]

    max_loss = np.max(loss)
    min_loss = np.min(loss)
    max_active = np.max(active)
    min_active = np.min(active)
    # If there aren't any values with active nodes, just say min_idx is 0
    if max_active == 0:
        best_idxs[node] = 0
        no_active.append(node)
        continue
    # Find the index of the model with the smallest number of active nodes and the smallest loss
    min_num_active = np.min(active[active>=1])
    min_num_actives.append(min_num_active)
    min_active = np.where(active == min_num_active)[0]
    min_idx = min_active[np.argmin(loss[min_active])]
    minimal_idxs[node] = min_idx
    
#%%
print('Nodes with no active nodes:')
for node in no_active:
    print(node)
# %%
# Assemble a combined model that has the best parameters for each node
trained_models = pickle.load(open(f'../../output/{tmstp}/models/resampled_group_l1_flow_models_{genotype}.pickle', 'rb'))
#%%
optimal_models = {}
for node in tqdm(nodes):
    idx = best_idxs[node]
    optimal_models[node] = trained_models[node][idx]
minimal_models = {}
for node in tqdm(nodes):
    if node in minimal_idxs:
        idx = minimal_idxs[node]
    else:
        idx = best_idxs[node]
    minimal_models[node] = trained_models[node][idx]
#%%
best_model = GroupL1FlowModel(input_dim=len(nodes),
                         hidden_dim=64,
                         num_layers=3,
                         predict_var=True)
for node in tqdm(optimal_models):
    best_model.models[node].load_state_dict(optimal_models[node])
# Save the optimal model
torch.save(best_model.state_dict(), 
           f'../../output/{tmstp}/models/optimal_{genotype}.torch')
#%%
minimal_model = GroupL1FlowModel(input_dim=len(nodes),
                         hidden_dim=64,
                         num_layers=3,
                         predict_var=True)
for node in tqdm(minimal_models):
    minimal_model.models[node].load_state_dict(minimal_models[node])
# Save the minimal model
torch.save(minimal_model.state_dict(), 
           f'../../output/{tmstp}/models/minimal_{genotype}.torch')
# %%
#%%
%load_ext autoreload
%autoreload 2
#%%
import plotting
#%%
best_model = best_model.to('cpu')

X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)
Vt = torch.tensor(V)
Xt = torch.tensor(X)

pcs = adata.varm['PCs']
pca = PCA()
pca.components_ = pcs.T
pca.mean_ = adata.X.mean(axis=0)

#%%
def embed(X, pcs=[0,1]):
    return pca.transform(X)[:,pcs]

#%%
embedding = embed(X)
V_emb = embed_velocity(X, V, embed)
#%%
starts = torch.tensor(X)
#%%
# Plot arrow grid showing the velocity vectors
plotting.arrow_grid(adata, pca, best_model, genotype, 'cpu', 
                    perturbation=None, true_velocities=Vt)

# %%
# MSE loss of prediction
loss = torch.nn.MSELoss()

best_pV = best_model(Xt)[0]
minimal_pV = minimal_model(Xt)[0]
print('Optimal model loss:', loss(best_pV, Vt).item())
print('Minimal model loss:', loss(minimal_pV, Vt).item())
# %%
