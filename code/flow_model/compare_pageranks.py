#%%
import pickle
import networkx as nx
import numpy as np
import random
from joblib import Parallel, delayed
import sys
sys.path.append('..')
from util import generate_random_shortest_paths

#%%
# Set the random seed
random.seed(0)
np.random.seed(0)
#%%
wt_tmstp = '20230607_165324'  
mut_tmstp = '20230608_093734'
outdir = f'../../output/'

protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {idx: '/'.join(name) for idx, name in protein_id_name.items()}

#%%
# Compute the difference in pagerank between the wildtype and mutant path graphs
wt_path_pr = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_pagerank_wildtype.pickle', 'rb'))
mut_path_pr = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_pagerank_mutant.pickle', 'rb'))

path_pr_diff = {node: wt_path_pr[node] - mut_path_pr[node]
                for node in wt_path_pr if node in mut_path_pr}

#%%
# Create random graphs with same degree distributions as the input graphs
# Then compute the shortest path graphs for all of them
# Then compute the pagerank differences between all pairs of graphs
# Finally compute the probabilities of the observed pagerank differences
# %%
# Load the graph
wt_selection_pvalues = pickle.load(open(f'{outdir}/{wt_tmstp}/input_selection_pvalues_wildtype.pickle', 'rb'))
mut_selection_pvalues = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle', 'rb'))
graph = pickle.load(open(f'../../data/filtered_graph.pickle','rb'))
node_list = list(graph.nodes)
#%%
# Generate shortest paths from *randomly selected* sources to each node
undirected_graph = graph.to_undirected()
num_random_graphs = 100

parallel = Parallel(n_jobs=30, verbose=10)
wt_random_shortest_path_graphs = parallel(delayed(generate_random_shortest_paths)(wt_selection_pvalues, undirected_graph)
                                          for _ in range(num_random_graphs));
mut_random_shortest_path_graphs = parallel(delayed(generate_random_shortest_paths)(mut_selection_pvalues, undirected_graph)
                                           for _ in range(num_random_graphs));
# %%
# Compute the pagerank of each node in each random graph
wt_random_path_pageranks = []
for path_graph in wt_random_shortest_path_graphs:
    wt_random_path_pageranks.append(nx.pagerank(path_graph.to_undirected(), weight='weight'))
mut_random_path_pageranks = []
for path_graph in mut_random_shortest_path_graphs:
    mut_random_path_pageranks.append(nx.pagerank(path_graph.to_undirected(), weight='weight'))

#%%
# Compute the difference in pagerank between each pair of random graphs
all_nodes = set(wt_random_path_pageranks[0]) & set(mut_random_path_pageranks[0])
random_path_pr_diffs = {node: [] for node in all_nodes}
for i in range(num_random_graphs):
    for j in range(num_random_graphs):
        for node in all_nodes:
            random_path_pr_diffs[node].append(wt_random_path_pageranks[i][node] - mut_random_path_pageranks[j][node])
#%%
def pct(node_diff, random_diffs, reverse=False):
    node, pr_diff = node_diff
    if reverse:
        return np.sum(pr_diff < np.array(random_diffs[node]))/len(random_diffs[node])
    else:
        return np.sum(pr_diff > np.array(random_diffs[node]))/len(random_diffs[node])

#%%
print('More PageRank central in wildtype')
from functools import partial
wt_pct = partial(pct, random_diffs=random_path_pr_diffs, reverse=False)
wt_path_pr_diff = sorted(path_pr_diff.items(), key=wt_pct, reverse=True)
for node, pr_diff in wt_path_pr_diff[:20]:
    # compute the percentile of the observed difference in the random distribution
    print(f'{protein_id_name[node]:8s}: {pr_diff:.5f} {wt_pct((node,pr_diff)):.5f}')
#%%
print('More PageRank central in mutant')
mut_pct = partial(pct, random_diffs=random_path_pr_diffs, reverse=True)
mut_path_pr_diff = sorted(path_pr_diff.items(), key=mut_pct, reverse=True)
for node, pr_diff in mut_path_pr_diff[:20]:
    print(f'{protein_id_name[node]:8s}: {pr_diff:.5f} {mut_pct((node,pr_diff)):.5f}')

#%%
mut_path_graph = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_graph_mutant.pickle', 'rb'))
wt_path_graph = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_graph_wildtype.pickle', 'rb'))
#%%
for node, pr in wt_path_pr_diff[:20]:
    one_hop_neighbs = set(nx.ego_graph(graph, node, radius=1).nodes())
    two_hop_neighbs = set(nx.ego_graph(graph, node, radius=2).nodes())
    print(protein_id_name[node], len(one_hop_neighbs), len(two_hop_neighbs))
#%%
for node, pr in mut_path_pr_diff[:20]:
    one_hop_neighbs = set(nx.ego_graph(graph, node, radius=1).nodes())
    two_hop_neighbs = set(nx.ego_graph(graph, node, radius=2).nodes())
    print(protein_id_name[node], len(one_hop_neighbs), len(two_hop_neighbs))
#%%
# Save the singificant nodes for both WT and mutant
wt_significant_nodes = [node for node, pr in wt_path_pr_diff if wt_pct((node,pr)) < 0.1]
mut_significant_nodes = [node for node, pr in mut_path_pr_diff if mut_pct((node,pr)) < 0.1]
pickle.dump(wt_significant_nodes, open(f'{outdir}/{wt_tmstp}/pagerank_diff_significant_wildtype.pickle', 'wb'))
pickle.dump(mut_significant_nodes, open(f'{outdir}/{mut_tmstp}/pagerank_diff_significant_mutant.pickle', 'wb'))
# %%
