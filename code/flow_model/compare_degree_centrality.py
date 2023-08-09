#%%
import pickle
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append('..')
from util import generate_random_shortest_paths
import random
#%%
# Set the random seed
random.seed(0)

#%%
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {idx: '/'.join(name) for idx, name in protein_id_name.items()}
#%%
# Load the shortest path graph for the wildtype and mutant
outdir = '../../output'
wt_tmstp = '20230607_165324'
wt_path_graph = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_graph_wildtype.pickle', 'rb'))
mut_tmstp = '20230608_093734'
mut_path_graph = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_graph_mutant.pickle', 'rb'))
# %%
wt_total_weight = sum([edge[2]['weight'] for edge in wt_path_graph.edges(data=True)])
mut_total_weight = sum([edge[2]['weight'] for edge in mut_path_graph.edges(data=True)])

pct_diffs = {}
all_nodes = set(wt_path_graph.nodes) & set(mut_path_graph.nodes)
for node in all_nodes:
    wt_edges = wt_path_graph.in_edges(node, data=True)
    mut_edges = mut_path_graph.in_edges(node, data=True)
    wt_edge_sum = sum([edge[2]['weight'] for edge in wt_edges])
    mut_edge_sum = sum([edge[2]['weight'] for edge in mut_edges])
    wt_pct = wt_edge_sum/wt_total_weight
    mut_pct = mut_edge_sum/mut_total_weight
    pct_diff = wt_pct - mut_pct
    pct_diffs[node] = pct_diff
#%%
# %%
# Load the graph
wt_selection_pvalues = pickle.load(open(f'{outdir}/{wt_tmstp}/input_selection_pvalues_wildtype.pickle', 'rb'))
mut_selection_pvalues = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle', 'rb'))
graph = pickle.load(open(f'../../data/filtered_graph.pickle','rb'))
#%%
# Generate shortest paths from *randomly selected* sources to each node
undirected_graph = graph.to_undirected()
num_random_graphs = 100

parallel = Parallel(n_jobs=30, verbose=10)
wt_random_shortest_path_graphs = parallel(delayed(generate_random_shortest_paths)(wt_selection_pvalues, undirected_graph)
                                          for _ in range(num_random_graphs));
mut_random_shortest_path_graphs = parallel(delayed(generate_random_shortest_paths)(mut_selection_pvalues, undirected_graph)
                                           for _ in range(num_random_graphs));

#%%
# Compare the degree centrality of all nodes 
# between all combinations of WT and mutant random shortest path graphs
random_diffs = {node: [] for node in all_nodes}
for wt_idx in range(num_random_graphs):
    for mut_idx in range(num_random_graphs):
        wt_rand_graph = wt_random_shortest_path_graphs[wt_idx]
        mut_rand_graph = mut_random_shortest_path_graphs[mut_idx]
        wt_total_weight = sum([edge[2]['weight'] for edge in wt_rand_graph.edges(data=True)])
        mut_total_weight = sum([edge[2]['weight'] for edge in mut_rand_graph.edges(data=True)])

        for node in all_nodes:
            wt_edges = wt_rand_graph.in_edges(node, data=True)
            mut_edges = mut_rand_graph.in_edges(node, data=True)
            wt_edge_sum = sum([edge[2]['weight'] for edge in wt_edges])
            mut_edge_sum = sum([edge[2]['weight'] for edge in mut_edges])
            wt_pct = wt_edge_sum/wt_total_weight
            mut_pct = mut_edge_sum/mut_total_weight
            pct_diff = wt_pct - mut_pct
            random_diffs[node].append(pct_diff)
#%%
for node, diffs in random_diffs.items():
    random_diffs[node] = np.array(diffs)
#%%
# Compute the pvalues of the observed differences
# compared to the random differences
pvalues = {}
for node in all_nodes:
    pvalues[node] = (np.sum(random_diffs[node] > pct_diffs[node])) / (num_random_graphs**2)


# %%
# Print the top 20 nodes with the largest difference in the percentage of flow
# that goes through them
sorted_diffs = sorted(pct_diffs.items(), key=lambda x: x[1], reverse=True)
#%%
print('Significant differences WT-Mut:')
for node, pct_diff in sorted_diffs:
    if pvalues[node] < 0.05:
        print(f'{protein_id_name[node]:8s}: {pct_diff:.4f} {pvalues[node]:.3f} *')
# Top 30 nodes with the largest difference in the percentage of flow overall, including significant and non-significant
print('Top 30 nodes WT-Mut, including non-significant differences:')
for node, pct_diff in sorted_diffs[:30]:
    print(f'{protein_id_name[node]:8s}: {pct_diff:.4f} {pvalues[node]:.3f}')
#%%
print('Significant differences Mut-WT:')
for node, pct_diff in sorted_diffs[::-1]:
    if 1-pvalues[node] < 0.05:
        print(f'{protein_id_name[node]:8s}: {pct_diff:.4f} {1-pvalues[node]:.3f} *')
print('Top 30 nodes Mut-WT, including non-significant differences:')
for node, pct_diff in sorted_diffs[::-1][:30]:
    print(f'{protein_id_name[node]:8s}: {pct_diff:.4f} {pvalues[node]:.3f}')
# %%
# Save the significant nodes
wt_significant_nodes = [node for node, pct_diff in sorted_diffs if pvalues[node] < 0.1]
pickle.dump(wt_significant_nodes, open(f'{outdir}/{wt_tmstp}/centrality_diff_wt.pickle', 'wb'))
mut_significant_nodes = [node for node, pct_diff in sorted_diffs[::-1] if 1-pvalues[node] < 0.1]
pickle.dump(mut_significant_nodes, open(f'{outdir}/{mut_tmstp}/centrality_diff_mut.pickle', 'wb'))