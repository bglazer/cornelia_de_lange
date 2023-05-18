#%%
import pickle
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
from flow_model import GraphFlowModel
from util import tonp, embed_velocity, plot_arrows, get_plot_limits, velocity_vectors
import scanpy as sc
from sklearn.decomposition import PCA
#%%
tmstp = '20230427_140640'
#%%
params = pickle.load(open(f'../output/{tmstp}/params/graph_flow_model_wildtype.pickle', 'rb'))

#%%
losses = {}
active_nodes = {}

n_params = len(params)
for i in range(n_params):
    logfiles = glob.glob(f'../output/{tmstp}/logs/*_{i}.log')
    for path in logfiles:
        filename = path.split('/')[-1]
        node = filename.split('_')[3]
        
        with open(path, 'r') as f:
            # print(filename)
            lines = f.readlines()
            if len(lines) == 0:
                continue
            if node not in losses:
                losses[node] = []
                active_nodes[node] = []
            line = lines[-1]
            sp = line.split()
            loss = float(sp[3])
            active = int(sp[6])
            losses[node].append(loss)
            active_nodes[node].append(active)
# %%
nodes = list(losses.keys())
best_idxs = {}
for node in nodes:
    # print(node)
    losses[node] = np.asarray(losses[node])
    active_nodes[node] = np.asarray(active_nodes[node])
    loss = losses[node]
    active = active_nodes[node]

    max_loss = np.max(loss)
    min_loss = np.min(loss)
    max_active = np.max(active)
    min_active = np.min(active)
    # If there aren't any values with active nodes, just say min_idx is 0
    if max_active == 0:
        best_idxs[node] = 0
        continue
    # If all values have the same number of active nodes, min_idx is the best loss
    if max_active == min_active:
        best_idxs[node] = np.argmin(loss)
        continue
    # Normalize losses and active nodes
    nrm_loss = (loss - min_loss) / (max_loss - min_loss)
    nrm_active = (active - min_active) / (max_active - min_active)
    # Remove indexes that have zero active nodes
    mask = active > 0
    nrm_loss = nrm_loss[mask]
    nrm_active = nrm_active[mask]
    # Find the value that minimizes loss and number of active nodes
    # i.e. min vector norm of the two normalized values
    norm = np.sqrt(nrm_loss**2 + nrm_active**2)
    # Find the index of the minimum norm
    min_idx = np.argmin(norm)
    best_idxs[node] = min_idx
    # Find the value of the parameter that minimizes the norm
    # print(min_idx, loss[min_idx], active[min_idx], params[min_idx])
    # plt.plot(loss, active, 'o')
    # plt.xlabel('Loss')
    # plt.ylabel('Active nodes')
    # plt.yticks(np.arange(min_active-1, max_active+1, 1))
    # # Add a red dot at the minimum norm
    # plt.plot(loss[min_idx], active[min_idx], 'ro')

# %%
# Assemble a combined model that has the best parameters for each node
trained_models = pickle.load(open(f'../output/{tmstp}/models/graph_flow_model_wildtype.pickle', 'rb'))
#%%
combined_model = {}
for node in nodes:
    idx = best_idxs[node]
    if node not in trained_models[idx]:
        print(f'Node {node} not in model {idx}')
    else:
        combined_model[node] = trained_models[idx][node]
# %%
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))
adata = sc.read_h5ad(f'../data/wildtype_net.h5ad')
model = GraphFlowModel(num_layers=3,
                       graph=graph, 
                       data_idxs=adata.uns['id_row'], 
                       hops=3)
# model.models = torch.nn.ModuleDict(combined_model)
model.models.load_state_dict(torch.nn.ModuleDict(trained_models[1]).state_dict())
model = model.to('cpu')

X = adata.X.toarray()
T = adata.obsm['transition_matrix']

V = velocity_vectors(T, X)

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
pV = torch.zeros_like(starts)

for node in model.models.keys():
    v = model(starts, node)
    # Get the index of the node
    idx = adata.uns['id_row'][node]
    pV[:,idx] = v
#%%
dpv = embed_velocity(X=tonp(starts),
                    velocity=tonp(pV),
                    embed_fn=embed)
#%%
idxs = np.arange(0, starts.shape[0], 1)
x_limits, y_limits = get_plot_limits(embedding)
plot_arrows(idxs=idxs,
            points=embedding, 
            V=V_emb,
            pV=dpv,
            sample_every=5,
            scatter=False,
            save_file=f'../figures/embedding/graph_vector_field_wildtype_{tmstp}.png',
            c=adata.obs['pseudotime'],
            s=1.5,
            xlimits=x_limits,
            ylimits=y_limits)
# %%
# MSE loss of prediction
loss = torch.nn.MSELoss()
print(loss(pV, torch.tensor(V)))