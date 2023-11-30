#%%
%load_ext autoreload
%autoreload 2
#%%
import pickle
import numpy as np
import pickle
import torch
import numpy as np
from mediators import find_mediators
import random
from joblib import Parallel, delayed
import os
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'

#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# %%
# genotype = 'wildtype'
# tmstp = '20230607_165324'
genotype = 'mutant'
tmstp = '20230608_093734'
outdir = f'../../output/{tmstp}'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}
graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))

#%%
# Load the target input list
with open(f'{outdir}/optimal_{genotype}_active_inputs.pickle', 'rb') as f:
    target_active_inputs = pickle.load(f)
# %%
from mediators import count_bridges, find_bridges
import networkx as nx
import scipy

# %%
sorted_nodes = sorted(node_to_idx, key=lambda x: node_to_idx[x])
g = nx.DiGraph()
g.add_edges_from(graph.edges)
sg = nx.to_scipy_sparse_array(g, nodelist=sorted_nodes)
D = scipy.sparse.csgraph.shortest_path(sg, directed=True, unweighted=True)
node_pairs = []
for target, active_inputs in target_active_inputs.items():
    tgt_idx = node_to_idx[target]
    for active_input in active_inputs:
        node_pairs.append((active_input, target))
model_bridges = count_bridges(node_pairs, D)    # %%

# %%
probs, bridges = find_bridges(target_active_inputs, graph, threshold=0.01, verbose=False)
#%%
print(len(probs))
for bridge, pairs in bridges.items():
    # print(protein_id_name[bridge])
    print(bridge)
    for src, dst in pairs:
        # print(protein_id_name[src], protein_id_name[dst])
        print(','.join([protein_id_name[n] for n in nx.shortest_path(graph, bridge, src)]))
        print(','.join([protein_id_name[n] for n in nx.shortest_path(graph, bridge, dst)]))
        print('-')
    print('-------')
# %%
