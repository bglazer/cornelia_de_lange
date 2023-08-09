#%%
import pickle
import scanpy as sc
import numpy as np
from tqdm import tqdm
import networkx as nx
from tabulate import tabulate
from IPython.display import HTML, display
import sys
sys.path.append('..')
from util import create_path_graph

#%%
# tmstp = '20230607_165324'  
# genotype = 'wildtype'
tmstp = '20230608_093734'
genotype = 'mutant'
outdir = f'../../output/{tmstp}'
# %%
# Load the mapping of indices to node names
adata = sc.read(f'../../data/{genotype}_net.h5ad')
from collections import Counter
gene_list = list(adata.var_names)
#%%
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
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
p_values = pickle.load(open(f'{outdir}/input_selection_pvalues_{genotype}.pickle', 'rb'))
#%%
# Rank genes by how often they're selected at a given p-value threshold
p_threshold = 0.01
gene_counts = Counter()
for model_idx in p_values:
    for gene_idx in p_values[model_idx]:
        if p_values[model_idx][gene_idx] < p_threshold:
            gene_counts[gene_idx] += 1
#%%
protein_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle','rb'))
#%%
header = ["Idx", "Gene", "Count", "Mean_Rank", "Var_Rank"]
table = []
rank = 0
last_count = 0
for gene, count in gene_counts.most_common():
    gene_idx = protein_to_idx[gene]
    if last_count > count:
        rank += 1
    # Get the index of the gene in the mean list
    mean_rank = np.where(sorted_mean_idxs == gene_idx)[0][0]
    var_rank = np.where(sorted_var_idxs == gene_idx)[0][0]
    table.append([rank, protein_id_name[gene], count, mean_rank, var_rank])
    last_count = count

display(HTML(tabulate(table, tablefmt='html', headers=header)))

#%%
# Output the gene list as a pickle object
pickle.dump(gene_counts, open(f'{outdir}/input_selection_table_{genotype}.pickle', 'wb'))

#%%
# Print the p_values of the genes
for model_idx in p_values:
    print('Model -',protein_id_name[model_idx])
    sorted_p_values = sorted(p_values[model_idx].items(), key=lambda x: x[1])
    for gene_idx, pval in sorted_p_values:
        # Print the pvalue with 5 decimal places but fill the space with blanks if the least significant digit is 0
        print(f'{protein_id_name[gene_idx]:8s} {pval:.3f} {"*" if pval < p_threshold else ""}')
    print('-')

#%%
shortest_paths = pickle.load(open(f'{outdir}/shortest_paths_{genotype}.pickle', 'rb'))
random_shortest_paths = pickle.load(open(f'{outdir}/random_shortest_paths_{genotype}.pickle', 'rb'))
#%%
# Create a mapping of node names to indices
idx_to_node = pickle.load(open(f'../../data/protein_id_to_idx.pickle','rb'))
node_to_idx = {v: k for k, v in idx_to_node.items()}

#%%
# Create a graph of all the shortest paths, weighted by the
# frequency of the transitions in the shortest paths
observed_path_graph = create_path_graph(shortest_paths)
# Create the graph of all the random shortest paths
random_path_graph = create_path_graph(random_shortest_paths)
pickle.dump(observed_path_graph, open(f'{outdir}/shortest_path_graph_{genotype}.pickle', 'wb'))
pickle.dump(random_path_graph, open(f'{outdir}/random_shortest_path_graph_{genotype}.pickle', 'wb'))

#%%
# Calculate the pagerank of each node in the graph
pagerank = nx.pagerank(observed_path_graph.to_undirected(), weight='weight')
randomized_pagerank = nx.pagerank(random_path_graph.to_undirected(), weight='weight')
ranked_pagerank = sorted(pagerank.items(),
                         key=lambda x:x[1]-randomized_pagerank[x[0]],
                         reverse=True)
for node, pr in ranked_pagerank[:10]:
    rpr = randomized_pagerank[node]
    print(f'{protein_id_name[node]:8s} {pr:.4f} {rpr:.4f}')
pickle.dump(pagerank, open(f'{outdir}/shortest_path_pagerank_{genotype}.pickle', 'wb'))
pickle.dump(randomized_pagerank, open(f'{outdir}/random_shortest_path_pagerank_{genotype}.pickle', 'wb'))
#%%
from scipy.stats import hypergeom
# Compute the probability of seeing each node in the random shortest paths 
# versus the actual shortest paths
path_node_pvals = {}
source_pvals = {}

#%%
# Organize shortest paths by source
shortest_path_srcs = {}
for target in shortest_paths:
    for path in shortest_paths[target]:
        src = path[0]
        if src not in shortest_path_srcs:
            shortest_path_srcs[src] = []
        shortest_path_srcs[src].append(path)
random_shortest_path_srcs = {}
for target in random_shortest_paths:
    for path in random_shortest_paths[target]:
        src = path[0]
        if src not in random_shortest_path_srcs:
            random_shortest_path_srcs[src] = []
        random_shortest_path_srcs[src].append(path)
#%%
for source in tqdm(shortest_path_srcs):
    observed_path_count = Counter()
    path_node_pvals[source] = {}

    # Compute the observed probability of seeing each node in the shortest paths
    for path in shortest_path_srcs[source]:
        for node in path[1:-1]:
            observed_path_count[node] += 1
    # For each node in the observed random paths
    # find the probability of seeing that node in the random shortest paths
    # both to the target and from the source
    src_path_count = Counter()
    # Compute the probability of seeing this node in the random paths to the target
    for random_path in random_shortest_path_srcs[source]:
        for node in random_path[1:-1]:
            src_path_count[node] += 1
    # Compute the probability of seeing this node in random paths from the source
    # sources = set([path[0] for path in shortest_paths[target]])
    # src_path_count = Counter()
    # for src in sources:
    #     for random_path in random_shortest_paths[src]:
    #         for node in random_path[1:-1]:
    #             src_path_count[node] += 1
    
    # Compute a hypergeometric enrichment test to check if the observed probability of 
    # seeing a node in a given number of shortest paths is significantly different
    # from the random probability of seeing that node N times in a selection from 1000 shortest paths
    # M overall population
    # n successes in population
    # N sample size
    # x successes in sample
    n_tests = 0
    for node, count in src_path_count.items():
        M = len(random_shortest_path_srcs[source]) # total number of shortest paths to target (random)
        n = count # number of random shortest paths to target that contain this node
        N = len(shortest_path_srcs[source]) # Number of observed shortest paths to target (learned from data)
        x = observed_path_count[node] # number of observed shortest paths to target that contain this node
        rv = hypergeom(M, n, N)
        pmf = rv.pmf(x)
        n_tests += 1
        source_pvals[(source, node)] = pmf
    # src_pvals = {}
    # n_tests = 0
    # for node, count in src_path_count.items():
    #     M = len(random_shortest_paths[target])*len(sources)
    #     n = count
    #     N = len(shortest_paths[target])
    #     rv = hypergeom(M, n, N)
    #     x = observed_path_count[node]
    #     pmf = rv.pmf(x)
    #     n_tests += 1
    #     src_pvals[(target, node)] = pmf
#%%
sig_threshold = .05/n_tests
sig_path_nodes = {source:[] for source in shortest_paths}
for (source, node), p in source_pvals.items():
    if p < sig_threshold:
        sig_path_nodes[source].append(node)
pickle.dump(sig_path_nodes, open(f'{outdir}/sig_path_nodes_{genotype}.pickle', 'wb'))
#%%
# Print the pvalues of the nodes
for source in path_node_pvals:
    print(protein_id_name[source])
    sorted_pvals = sorted(path_node_pvals[source].items(), 
                          # sort by the combined difference in pvalues 
                          key=lambda x: (x[1][0] - x[1][1]) + (x[1][0] - x[1][2]), 
                          reverse=True)
    for node, pvals in sorted_pvals:
        p, tgt_p, src_p = pvals
        over = '*' if p > tgt_p  and p > src_p else ''
        print(f'{protein_id_name[node]:<8} {p:.3f} {tgt_p:.3f} {src_p:.3f} {over}')
    print('-')

with open(f'{outdir}/shortest_path_node_pvalues_{genotype}.pickle', 'wb') as f:
    pickle.dump(path_node_pvals, f)

#%%
# Find the genes that are most commonly overrepresented in the shortest paths
# versus the random shortest paths
all_path_node_pct_diffs = []
# Across all targets and nodes, find the 99th percentile of the difference in the
# percentage of times a node is in the observed shortest paths versus 
# the shortest paths from random nodes (null model)
for dest in path_node_pvals:
    for path_node in path_node_pvals[dest]:
        observed_pct, tgt_pct, src_pct = path_node_pvals[dest][path_node]
        if observed_pct > tgt_pct and observed_pct > src_pct:
            all_path_node_pct_diffs.append((observed_pct - tgt_pct)+(observed_pct - src_pct))
all_path_node_pct_diffs = np.array(all_path_node_pct_diffs)
pct_diff_threshold = np.percentile(all_path_node_pct_diffs, 90)

# Find the nodes that are most overrepresented in the shortest paths
overrepresented_nodes = Counter()
overrepresented_node_paths = {}
for source in path_node_pvals:
    for node in path_node_pvals[source]:
        observed_pct, tgt_pct, src_pct = path_node_pvals[source][node]
        if observed_pct > tgt_pct and observed_pct > src_pct:
            overrepresented_nodes[node] += 1
            if node not in overrepresented_node_paths:
                overrepresented_node_paths[node] = []
            overrepresented_node_paths[node] += [path for path in shortest_paths[source]
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