#%%
import networkx as nx
import pickle
from glob import glob
import json
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt
import torch
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from tabulate import tabulate
from IPython.display import HTML, display

#%%
# Set the random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#%%
tmstp = '20230607_165324'  
genotype = 'wildtype'
outdir = f'../../output/{tmstp}'
#%%
# Load the saved models
models = pickle.load(open(f'{outdir}/models/resampled_group_l1_flow_models_{genotype}.pickle', 'rb'))

#%%
# Parse the logs
# %%
logfiles = glob(f'../../output/{tmstp}/logs/*.log')
# %%
logs = {}
for file in logfiles:
    model_idx = int(file.split('/')[-1].split('_')[3])
    logs[model_idx] = []
    with open(file, 'r') as f:
        for line in f:
            log = json.loads(line)
            logs[model_idx].append(log)

# %%
# Load the mapping of indices to node names
adata = sc.read(f'../../data/{genotype}_net.h5ad')

#%%
# Find the final validation loss for each model
validation_loss = {}

for model_idx in logs:
    validation_loss[model_idx] = []
    for log in logs[model_idx]:
        if 'iteration' in log:
            validation_loss[model_idx].append(log['validation_loss'])


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
# %%
# Find the nodes that are active in the best models
active_idxs = {}

for model_idx in logs:
    active_idxs[model_idx] = []
    for model in models[model_idx]:
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
pickle.dump(p_values, open(f'{outdir}/input_selection_pvalues_{genotype}.pickle', 'wb'))
#%%
# Genes with highest variance in the data
variance = np.var(adata.X.toarray(), axis=0)
variance = np.array(variance).flatten()
sorted_var_idxs = np.argsort(variance)[::-1]
for gene_idx in sorted_var_idxs[:10]:
    print(f'{gene_idx:4d} {gene_list[gene_idx]:8s} {variance[gene_idx]:.3f}')
#%%
# Genes with highest mean in the data
mean = np.mean(adata.X.toarray(), axis=0)
mean = np.array(mean).flatten()
sorted_mean_idxs = np.argsort(mean)[::-1]
for gene_idx in sorted_mean_idxs[:10]:
    print(f'{gene_idx:4d} {gene_list[gene_idx]:8s} {mean[gene_idx]:.3f}')

#%%
# Rank genes by how often they're selected at a given p-value threshold
p_threshold = 0.01
gene_counts = Counter()
for model_idx in p_values:
    for gene_idx in p_values[model_idx]:
        if p_values[model_idx][gene_idx] < p_threshold:
            gene_counts[gene_idx] += 1
#%%
header = ["Idx", "Gene", "Count", "Mean_Rank", "Var_Rank"]
table = []
rank = 0
last_count = 0
for gene_idx, count in gene_counts.most_common():
    if last_count > count:
        rank += 1
    # Get the index of the gene in the mean list
    mean_rank = np.where(sorted_mean_idxs == gene_idx)[0][0]
    var_rank = np.where(sorted_var_idxs == gene_idx)[0][0]
    table.append([rank, gene_list[gene_idx], count, mean_rank, var_rank])
    last_count = count

display(HTML(tabulate(table, tablefmt='html', headers=header)))

#%%
# Output the gene list as a pickle object
pickle.dump(gene_counts, open(f'{outdir}/input_selection_table_{genotype}.pickle', 'wb'))

#%%
# Print the p_values of the genes
for model_idx in p_values:
    print('Model -',gene_list[model_idx])
    sorted_p_values = sorted(p_values[model_idx].items(), key=lambda x: x[1])
    for gene_idx, pval in sorted_p_values:
        # Print the pvalue with 5 decimal places but fill the space with blanks if the least significant digit is 0
        print(f'{gene_list[gene_idx]:8s} {pval:.3f} {"*" if pval < p_threshold else ""}')
    print('-')

# %%
# Find the shortest paths from the active nodes to the output
shortest_paths = {}
shortest_path_lens = {}
# Make the graph undirected
undirected_graph = graph.to_undirected()
for model_idx in active_idxs:
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
# Generate shortest paths from *randomly selected* inputs
def generate_random_shortest_paths(model_idx):
    target = idx_to_node[model_idx]
    random_shortest_paths = []
    for iter in range(num_random_iters):
        random_sources = random.choices(node_list, k=len(selection_counts[model_idx]))
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

#%%
# Compute the probability of seeing each node in the random shortest paths 
# versus the actual shortest paths
path_node_pvals = {}
for target in tqdm(shortest_paths):
    all_path_nodes = Counter()
    # Compute the observed probability of seeing each node in the shortest paths
    for path in shortest_paths[target]:
        for node in path[1:-1]:
            all_path_nodes[node] += 1
        
    path_node_pvals[target] = {}
    # Compute the probability of seeing this node in the random paths
    for node in all_path_nodes:
        random_count = 0
        for random_path in random_shortest_paths[target]:
            if node in random_path[1:-1]:
                random_count += 1
                
        random_p = random_count / len(random_shortest_paths[target])
        p = all_path_nodes[node] / len(shortest_paths[target])

        path_node_pvals[target][node] = (p, random_p)

#%%
# Print the pvalues of the nodes
for target in path_node_pvals:
    print("/".join(protein_id_name[target]))
    sorted_pvals = sorted(path_node_pvals[target].items(), key=lambda x: x[1][0] - x[1][1], reverse=True)
    for node, pvals in sorted_pvals:
        p, random_p = pvals
        print(f'{"/".join(protein_id_name[node]):<8} {p:.3f} {random_p:.3f} {"*" if p > random_p else ""}')
    print('-')

pickle.dump(path_node_pvals, open(f'{outdir}/shortest_path_node_pvalues_{genotype}.pickle', 'wb'))

#%%
# Find the genes that are most commonly overrepresented in the shortest paths
# versus the random shortest paths
all_path_node_pct_diffs = []
# Across all targets and nodes, find the 99th percentile of the difference in the
# percentage of times a node is in the observed shortest paths versus 
# the shortest paths from random nodes (null model)
for dest in path_node_pvals:
    for path_node in path_node_pvals[dest]:
        observed_pct, random_pct = path_node_pvals[dest][path_node]
        all_path_node_pct_diffs.append(observed_pct - random_pct)
all_path_node_pct_diffs = np.array(all_path_node_pct_diffs)
pct_diff_threshold = np.percentile(all_path_node_pct_diffs, 90)

# Find the nodes that are most overrepresented in the shortest paths
overrepresented_nodes = Counter()
overrepresented_node_paths = {}
for target in path_node_pvals:
    for node in path_node_pvals[target]:
        p, random_p = path_node_pvals[target][node]
        if p - random_p > pct_diff_threshold:
            overrepresented_nodes[node] += 1
            if node not in overrepresented_node_paths:
                overrepresented_node_paths[node] = []
            overrepresented_node_paths[node] += [path for path in shortest_paths[target]
                                                 if node in path[1:-1]]

#%%
header = ["Idx", "Gene", "Count", "Mean_Rank", "Var_Rank"]
table = []
rank = 0
last_count = 0
for node, count in overrepresented_nodes.most_common():
    if node not in node_to_idx:
        continue
    if last_count > count:
        rank += 1
    gene_idx = node_to_idx[node]
    # Get the index of the gene in the mean list
    mean_rank = np.where(sorted_mean_idxs == gene_idx)[0][0]
    var_rank = np.where(sorted_var_idxs == gene_idx)[0][0]
    table.append([rank, gene_list[gene_idx], count, mean_rank, var_rank])
    last_count = count

display(HTML(tabulate(table, tablefmt='html', headers=header)))

#%%
# Output the gene list as a pickle object
pickle.dump(overrepresented_nodes, open(f'{outdir}/shortest_path_table_{genotype}.pickle', 'wb'))
pickle.dump(overrepresented_node_paths, open(f'{outdir}/overrepresented_node_paths_{genotype}.pickle', 'wb'))

# %%
# for model_idx in shortest_paths:
#     for (src,dst), pathlen in shortest_path_lens[model_idx].items():
#         print(f"{'/'.join(protein_id_name[src]):<8}", 
#               f"{'/'.join(protein_id_name[dst]):<8}", 
#               pathlen)

