#%%
import networkx as nx
import pickle
import scanpy as sc
import numpy as np
import torch
import random
from tqdm import tqdm
from joblib import Parallel, delayed

#%%
# Set the random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#%%
# tmstp = '20230607_165324'  
# genotype = 'wildtype'
tmstp = '20230608_093734'
genotype = 'mutant'
outdir = f'../../output/{tmstp}'
#%%
# Load the saved models
models = pickle.load(open(f'{outdir}/models/resampled_group_l1_flow_models_{genotype}.pickle', 'rb'))
# %%
# Load the mapping of indices to node names
adata = sc.read(f'../../data/{genotype}_net.h5ad')

# %%
# Load the graph
graph = pickle.load(open(f'../../data/filtered_graph.pickle','rb'))
# %%
# Load the mapping of gene names to ensembl ids
protein_name_id = pickle.load(open(f'../../data/protein_names.pickle','rb'))
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
# %%
idx_to_node = {}
node_to_idx = {}
for i,protein_name in enumerate(adata.var_names):
    if protein_name not in protein_name_id:
        print(protein_name, ' not mapped to id')
    for protein_id in protein_name_id[protein_name]:
        if protein_id in graph.nodes:
            idx_to_node[i] = protein_id
            node_to_idx[protein_id] = i
            break

print(len(idx_to_node), ' nodes mapped to indices')
print(len(adata.var_names), ' total nodes')    
pickle.dump(node_to_idx, open(f'../../data/protein_id_to_idx.pickle', 'wb'))
pickle.dump(idx_to_node, open(f'../../data/protein_idx_to_id.pickle', 'wb'))
# %%
# Find the nodes that are active in the best models
active_idxs = {}

for model_idx, node_models in enumerate(models):
    active_idxs[model_idx] = []
    for model in node_models:
        # Get the indices of the active inputs from the weight mask
        idxs = [int(idx) for idx in torch.where(model['group_l1'] > 0)[0]]
        active_idxs[model_idx].append(idxs)
        
# %%
# Find the most common active inputs
from collections import Counter
gene_list = list(adata.var_names)
node_list = list(idx_to_node.values())
num_nodes = len(node_list)
selection_counts = {}
for model_idx in active_idxs:
    selection_counts[model_idx] = Counter()
    for sample in active_idxs[model_idx]:
        selection_counts[model_idx].update(sample)

#%%
# Randomly select sets of nodes the same size as the active nodes for each model/iteration
random_selection_counts = {}
num_random_iters = 1000
for model_idx in tqdm(active_idxs):
    random_selection_counts[model_idx] = []
    for iter in range(num_random_iters):
        c = Counter()
        for sample in active_idxs[model_idx]:
            # Randomly sample the same number of genes
            random_sample = random.choices(range(num_nodes), k=len(sample))
            c.update(random_sample)
        random_selection_counts[model_idx].append(c)
#%%
# Calculate the percentage of times each gene was selected randomly versus how many 
# times it was selected in the actual model
p_values = {}
for model_idx in tqdm(selection_counts):
    model_selection_counts = selection_counts[model_idx]
    p_values[model_idx] = {}
    for gene_idx in model_selection_counts:
        random_count = 0
        for random_counter in random_selection_counts[model_idx]:
            if gene_idx in random_counter and random_counter[gene_idx] >= model_selection_counts[gene_idx]:
                random_count += 1
        p_values[model_idx][gene_idx] = random_count / num_random_iters
# TODO repeat this with pairs?
#%%
with open(f'{outdir}/input_selection_pvalues_{genotype}.pickle', 'wb') as pvalfile:
    converted_p_values = {}
    for model_idx in p_values:
        converted_p_values[idx_to_node[model_idx]] = {idx_to_node[gene_idx]: pval 
                                                      for gene_idx, pval 
                                                      in p_values[model_idx].items()}
    pickle.dump(converted_p_values, pvalfile)
# %%
# Find the shortest paths from the active nodes to the output
shortest_paths = {}
shortest_path_lens = {}
# Make the graph undirected
undirected_graph = graph.to_undirected()
for model_idx in tqdm(active_idxs):
    target = idx_to_node[model_idx]
    shortest_paths[target] = []
    shortest_path_lens[target] = {}
    for source_idx in selection_counts[model_idx]:
        source = node_list[source_idx]
        try:
            paths = list(nx.all_shortest_paths(undirected_graph, source, target))
            shortest_paths[target] += paths
            shortest_path_lens[target][source] = len(paths[0])
        except nx.NetworkXNoPath:
            pass
#%%
pickle.dump(shortest_paths, open(f'{outdir}/shortest_paths_{genotype}.pickle', 'wb'))
#%%
targets = {}
for model_idx in selection_counts:
    for input_idx in selection_counts[model_idx]:
        if input_idx not in targets:
            targets[input_idx] = set()
        targets[input_idx].add(model_idx)
#%%
# Generate shortest paths from *randomly selected* inputs
num_random_iters = 1000
def generate_random_shortest_paths(model_idx):
    target = idx_to_node[model_idx]
    random_shortest_paths = []
    random_sources = random.choices(node_list, k=num_random_iters)
    for source in random_sources:
        try:
            paths = list(nx.all_shortest_paths(undirected_graph, source, target))
            random_shortest_paths += paths
        except nx.NetworkXNoPath:
            pass
    return (target, random_shortest_paths)

parallel = Parallel(n_jobs=30, verbose=10)
random_shortest_paths = parallel(delayed(generate_random_shortest_paths)(model_idx)
                                 for model_idx in active_idxs);
random_shortest_paths = dict(random_shortest_paths)
pickle.dump(random_shortest_paths, open(f'{outdir}/random_shortest_paths_{genotype}.pickle', 'wb'))
# %%
