#%%
%load_ext autoreload
%autoreload 2
#%%
from flow_model import GraphFlowModel
import networkx as nx
import pickle
import torch
import scanpy as sc
from torch.optim import Adam
 
#%%
genotype = 'wildtype'
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))
adata = sc.read_h5ad('../data/wildtype_net.h5ad')
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_synonyms.pickle', 'rb'))
 
 #%%
nodes = list(adata.var_names)
indices_of_nodes_in_graph = []
node_idxs = {}

# Get the indexes of the data rows that correspond to nodes in the Nanog regulatory network
for i,name in enumerate(nodes):
    name = name.upper()
    # Find the ensembl id of the gene
    if name in protein_name_to_ids:
        # There may be multiple ensembl ids for a gene name
        for id in protein_name_to_ids[name]:
            # If the ensembl id is in the graph, then the gene is in the network
            if id in graph.nodes:
                # Record the data index of the gene in the network data
                node_idxs[id] = i
#%%
# Get the indexes of the data rows that correspond to nodes in the Nanog regulatory network
model = GraphFlowModel(num_layers=2, graph=graph, data_idxs=node_idxs, hops=2).to('cuda:0')

#%%
x = torch.tensor(adata.X.toarray(), device='cuda:0')
y = model(x, 0)
# input is num_samples x num_nodes
optimizer = Adam(model.parameters(), lr=1e-3)
# breakpoint()
 # %%
num_nodes = len(node_idxs)
y = torch.zeros((42, num_nodes), device='cuda:0')
loss = torch.nn.MSELoss(reduction='mean')
for i in range(500):
    optimizer.zero_grad()
    yhat = model(torch.zeros((42, num_nodes), device='cuda:0'))
    l=loss(yhat, y)
    l.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, l.item(), flush=True)

print(model.models[0].l1, flush=True)

breakpoint()

# %%
