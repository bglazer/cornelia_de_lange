#%%
# Automatically reload modules that are changed externally
# %load_ext autoreload
# %autoreload 2
#%%
from flow_model import GraphFlowModel
import networkx as nx
import pickle
import torch
import scanpy as sc
from torch.optim import Adam
import numpy as np

#%% 
# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

#%%
genotype = 'wildtype'
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))
adata = sc.read_h5ad('../data/wildtype_net.h5ad')

 #%%
nodes = list(adata.var_names)
#%%
# Get the indexes of the data rows that correspond to nodes in the Nanog regulatory network
model = GraphFlowModel(num_layers=2, 
                       graph=graph, 
                       data_idxs=adata.uns['id_row'], 
                       hops=3).to('cuda:0')

#%%
x = torch.tensor(adata.X.toarray(), device='cuda:0')
node = list(graph.nodes)[0]
y = model(x, node)
# input is num_samples x num_nodes
optimizer = Adam(model.parameters(), lr=1e-3)
# breakpoint()
 # %%
num_nodes = len(node_idxs)
y = torch.zeros((x.shape[0], 1), device='cuda:0')
loss = torch.nn.MSELoss(reduction='mean')
for i in range(500):
    optimizer.zero_grad()
    yhat = model(x, node)
    l=loss(yhat, y)
    l.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, l.item(), flush=True)

# print(model.models[0].l1, flush=True)

breakpoint()

# %%
