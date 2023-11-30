#%%
import torch
import pickle
import scanpy as sc
import networkx as nx
from tqdm import tqdm

#%%
# genotype = 'wildtype'
# tmstp = '20230607_165324'
genotype = 'mutant'
tmstp = '20230608_093734'
mut = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
cell_types = {c:i for i,c in enumerate(set(mut.obs['cell_type']))}
outdir = f'../../output/{tmstp}'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {idx: node for node, idx in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}

#%%
# Load the optimal model
state_dict = torch.load(f'../../output/{tmstp}/models/optimal_{genotype}.torch')
# %%
target_l1s = {}
for key in state_dict.keys():
    if 'group_l1' in key:
        target = key.split('.')[1]
        target_l1s[target] = state_dict[key]
# %%
target_active_idxs = {}
for target,l1s in target_l1s.items():
    # Get the indices of the entries that are greater than zero
    active_idxs = torch.argwhere(l1s>0).squeeze()
    target_active_idxs[target] = active_idxs
# %%
# Convert the indices to protein ids
target_active_genes = {}
for target,active_idxs in target_active_idxs.items():
    target_id = idx_to_node[int(target)]
    if len(active_idxs.shape) > 0:
        active_ids = [idx_to_node[int(idx)] for idx in active_idxs]
        target_active_genes[target_id] = active_ids
    else:
        target_active_genes[target_id] = []
# %%
# Save the active genes
with open(f'../../output/{tmstp}/optimal_{genotype}_active_inputs.pickle', 'wb') as f:
    pickle.dump(target_active_genes, f)
# %%
# Create a graph of the active genes
graph = nx.DiGraph()
for target_gene, active_genes in target_active_genes.items():
    for active_gene in active_genes:
        graph.add_edge(active_gene, target_gene)
# %%
# Calculate the number of connected components
print(f'Number of connected components: {nx.number_weakly_connected_components(graph)}')
# %%
# Save the optimal model graph
pickle.dump(graph, open(f'../../output/{tmstp}/optimal_{genotype}_graph.pickle', 'wb'))
# %%
regulatory_graph = pickle.load(open(f'../../data/filtered_graph.pickle','rb'))
#%%
# Find the shortest paths from the active nodes to the target nodes
shortest_paths = {}
# Make the graph undirected
undirected_regulatory_graph = regulatory_graph.to_undirected()
for target, sources in tqdm(target_active_genes.items()):
    shortest_paths[target] = []
    for source in sources:
        try:
            paths = list(nx.all_shortest_paths(undirected_regulatory_graph, source, target))
            shortest_paths[target] += paths
        except nx.NetworkXNoPath:
            pass
#%%
# All pairs shortest paths
all_shortest_paths = {}
for target in tqdm(node_to_idx):
    all_shortest_paths[target] = []
    for source in node_to_idx:
        try:
            paths = list(nx.all_shortest_paths(undirected_regulatory_graph, source, target))
            all_shortest_paths[target] += paths
        except nx.NetworkXNoPath:
            pass
#%%
# Save the all pairs shortest paths
with open(f'../../output/{tmstp}/all_shortest_paths.pickle', 'wb') as f:
    pickle.dump(all_shortest_paths, f)
#%%
# Create an optimal model shortest paths graph
optimal_model_shortest_paths_graph = nx.DiGraph()
for target, paths in shortest_paths.items():
    for path in paths:
        nx.add_path(optimal_model_shortest_paths_graph, path)
# %%
# Save the optimal model shortest paths graph and the shortest paths dictionary
pickle.dump(optimal_model_shortest_paths_graph, open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths_graph.pickle', 'wb'))
pickle.dump(shortest_paths, open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths.pickle', 'wb'))


# %%
