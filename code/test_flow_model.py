#%%
from flow_model import FlowModel
import networkx as nx
import pickle
import torch

#%%
genotype = 'wildtype'
graph = pickle.load(open(f'../data/filtered_graph.pickle', 'rb'))
network_data = pickle.load(open(f'../data/network_data_{genotype}.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_synonyms.pickle', 'rb'))

#%%
nodes = list(network_data.var_names)
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
flow_model = FlowModel(num_layers=2, graph=graph, data_idxs=node_idxs).to('cuda:0')

#%%
y = flow_model(torch.tensor(network_data.X, device='cuda:0'))
from torch.optim import Adam
optimizer = Adam(flow_model.parameters(), lr=1e-3)
breakpoint()
# %%
