#%%
import pickle
import scanpy as sc
import numpy as np
import scanpy as sc
# import numpy as np
import pickle
import torch
import sys
sys.path.append('..')
import numpy as np
from tabulate import tabulate
import os
#%%
# Set the random seed
np.random.seed(0)
torch.manual_seed(0)
#%%
os.environ['LD_LIBRARY_PATH'] = '/home/bglaze/miniconda3/envs/cornelia_de_lange/lib/'
# %%
wt_tmstp = '20230607_165324'
mut_tmstp = '20230608_093734'
wt_outdir = f'../../output/{wt_tmstp}'
mut_outdir = f'../../output/{mut_tmstp}'
wt_ko_dir = f'{wt_outdir}/knockout_simulations'
wt_pltdir = f'{wt_outdir}/knockout_simulations/figures'
mut_ko_dir = f'{mut_outdir}/knockout_simulations'
mut_pltdir = f'{mut_outdir}/knockout_simulations/figures'
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')
cell_types = {c:i for i,c in enumerate(set(wt_data.obs['cell_type']))}

#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}
#%%
# Load the mediated interactions
wt_mediated_interactions = pickle.load(open(f'{wt_outdir}/mediated_interactions_wildtype.pickle', 'rb'))
mut_mediated_interactions = pickle.load(open(f'{mut_outdir}/mediated_interactions_mutant.pickle', 'rb'))
#%%
wt_cell_type_ko_proportions = {}
for i, mediator in enumerate(wt_mediated_interactions):
    mediator_gene_name = protein_id_name[mediator]
    # Load the knockout results
    with open(f'{wt_ko_dir}/data/{mediator_gene_name}_wildtype_mediator_knockout_cell_type_proportions.pickle', 'rb') as f:
        wt_ko_cell_type_proportions = pickle.load(f)
        wt_perturb_cell_proportions, wt_baseline_cell_proportions = wt_ko_cell_type_proportions

        wt_cell_type_ko_proportions[mediator] = {}
        for i,cell_type in enumerate(cell_types):
            wt_cell_type_ko_proportions[mediator][cell_type] = wt_perturb_cell_proportions[i]

mut_cell_type_ko_proportions = {}
for i, mediator in enumerate(mut_mediated_interactions):
    mediator_gene_name = protein_id_name[mediator]
    # Load the knockout results
    with open(f'{mut_ko_dir}/data/{mediator_gene_name}_mutant_mediator_knockout_cell_type_proportions.pickle', 'rb') as f:
        mut_ko_cell_type_proportions = pickle.load(f)
        mut_perturb_cell_proportions, mut_baseline_cell_proportions = mut_ko_cell_type_proportions

        mut_cell_type_ko_proportions[mediator] = {}
        for i,cell_type in enumerate(cell_types):
            mut_cell_type_ko_proportions[mediator][cell_type] = mut_perturb_cell_proportions[i]
# %%
# Sort by the difference in means
wt_cell_type_changes = {}
for mediator, proportions in wt_cell_type_ko_proportions.items():
    cell_type_array = np.array([proportions[cell_type] for cell_type in cell_types])
    cell_type_diffs = cell_type_array - wt_baseline_cell_proportions
    cell_type_diff_sum = np.sum(np.abs(cell_type_diffs))
    wt_cell_type_changes[mediator] = (cell_type_diff_sum, cell_type_diffs)

mut_cell_type_changes = {}
for mediator, proportions in mut_cell_type_ko_proportions.items():
    cell_type_array = np.array([proportions[cell_type] for cell_type in cell_types])
    cell_type_diffs = cell_type_array - mut_baseline_cell_proportions
    cell_type_diff_sum = np.sum(np.abs(cell_type_diffs))
    mut_cell_type_changes[mediator] = (cell_type_diff_sum, cell_type_diffs)
#%%
# Compute difference in knockout changes in WT and mutant
diffs = {}
all_mediators = set(wt_mediated_interactions.keys()).union(set(mut_mediated_interactions.keys()))

for mediator in all_mediators:
    if mediator in wt_cell_type_changes:
        wt_cell_type_diff_sum, _ = wt_cell_type_changes[mediator]
    else:
        wt_cell_type_diff_sum = 0
    if mediator in mut_cell_type_changes:
        mut_cell_type_diff_sum, _ = mut_cell_type_changes[mediator]
    else:
        mut_cell_type_diff_sum = 0
    diffs[mediator] = wt_cell_type_diff_sum - mut_cell_type_diff_sum
#%%
# Sort by the difference in knockout changes in WT and mutant
sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

#%%
# Genes with highest variance in the data
wt_variance = np.var(wt_data.X.toarray(), axis=0)
wt_variance = np.array(wt_variance).flatten()
wt_sorted_var_idxs = np.argsort(wt_variance)[::-1]
# Genes with highest mean in the data
wt_mean = np.mean(wt_data.X.toarray(), axis=0)
wt_mean = np.array(wt_mean).flatten()
wt_sorted_mean_idxs = np.argsort(wt_mean)[::-1]

# Genes with highest variance in the data
mut_variance = np.var(mut_data.X.toarray(), axis=0)
mut_variance = np.array(mut_variance).flatten()
mut_sorted_var_idxs = np.argsort(mut_variance)[::-1]
# Genes with highest mean in the data
mut_mean = np.mean(mut_data.X.toarray(), axis=0)
mut_mean = np.array(mut_mean).flatten()
mut_sorted_mean_idxs = np.argsort(mut_mean)[::-1]
#%%
# Get the list of robust mediators
wt_robust_mediators = pickle.load(open(f'{wt_outdir}/robust_mediators_wildtype.pickle', 'rb'))
mut_robust_mediators = pickle.load(open(f'{mut_outdir}/robust_mediators_mutant.pickle', 'rb'))
#%%
wt_mean_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(wt_sorted_mean_idxs)}
wt_var_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(wt_sorted_var_idxs)}
mut_mean_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(mut_sorted_mean_idxs)}
mut_var_sorted_genes = {idx_to_node[idx]: i for i,idx in enumerate(mut_sorted_var_idxs)}

#%%
# Write a table
# Print the table
rows = []
relevant_mediators = []
headers = ['Mediator ID', "Mediator", "WT Cell Type Change", "Mutant Cell Type Change", "Difference", 
           'WT/Mut Mean Rank', 'WT/Mut Variance Rank', 'WT Robust', 'Mutant Robust']
for mediator, diff in sorted_diffs:
    # Get the list of genes that are significantly mediated by this mediator
    mediator_gene = protein_id_name[mediator]
    if mediator in wt_cell_type_changes:
        wt_cell_type_diff_sum, _ = wt_cell_type_changes[mediator]
    else:
        wt_cell_type_diff_sum = 0
    if mediator in mut_cell_type_changes:
        mut_cell_type_diff_sum, _ = mut_cell_type_changes[mediator]
    else:
        mut_cell_type_diff_sum = 0
    
    if mediator in wt_mean_sorted_genes:
        wt_mean_rank = f'{wt_mean_sorted_genes[mediator]}'
        wt_var_rank = f'{wt_var_sorted_genes[mediator]}'
    else:
        wt_mean_rank = 'NA'
        wt_var_rank = 'NA'
    if mediator in mut_mean_sorted_genes:
        mut_mean_rank = f'{mut_mean_sorted_genes[mediator]}'
        mut_var_rank = f'{mut_var_sorted_genes[mediator]}'
    else:
        mut_mean_rank = 'NA'
        mut_var_rank = 'NA'

    # if (diff > 0 and mediator in wt_robust_mediators) or (diff < 0 and mediator in mut_robust_mediators):
    relevant_mediators.append(mediator)
    rows.append([
        f'{mediator:10s}',
        f'{protein_id_name[mediator]:10s}',
        f'{wt_cell_type_diff_sum:4f}',
        f'{mut_cell_type_diff_sum:4f}',
        f'{diff:4f}',
        f'{wt_mean_rank}/{mut_mean_rank}',
        f'{wt_var_rank}/{mut_var_rank}',
        f'{mediator in wt_robust_mediators}',
        f'{mediator in mut_robust_mediators}'
    ])

print(tabulate(rows, headers=headers))


# %%
# Print the mediated interactions for each mediator
for mediator, diff in sorted_diffs:
    mediator_gene = protein_id_name[mediator]
    print(f'{mediator_gene:10s} {diff: 4f}')
    if mediator in wt_mediated_interactions:
        wt_mediated = wt_mediated_interactions[mediator]
    else:
        wt_mediated = []
    if mediator in mut_mediated_interactions:
        mut_mediated = mut_mediated_interactions[mediator]
    else:
        mut_mediated = []

    all_mediated = set(wt_mediated).union(set(mut_mediated))
    for src_dst in sorted(all_mediated):
        src, dst = src_dst
        src_gene = protein_id_name[src]
        dst_gene = protein_id_name[dst]
        print(f'    {src_gene:10s} -> {dst_gene:10s} '
              f'WT: {"y" if src_dst in wt_mediated else "n"} Mut: {"y" if src_dst in mut_mediated else "n"}')
# %%
# #%%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# #%%
# from plotting import optimize_placement, plot_paths
# import matplotlib.pyplot as plt
# import networkx as nx
# #%%

# for mediator, diff in sorted_diffs[1:2]:
#     mediator_gene = protein_id_name[mediator]
#     print(f'{mediator_gene:10s} {diff: 4f}')
#     if mediator in wt_mediated_interactions:
#         wt_mediated = wt_mediated_interactions[mediator]
#     else:
#         wt_mediated = []
#     if mediator in mut_mediated_interactions:
#         mut_mediated = mut_mediated_interactions[mediator]
#     else:
#         mut_mediated = []

#     all_mediated = set(wt_mediated).union(set(mut_mediated))
#     all_paths = []
#     for src, dst in sorted(all_mediated):
#         for path in nx.shortest_paths.all_shortest_paths(graph.to_undirected(), src, dst):
#             if mediator in path:
#                 all_paths.append(path)
#     # Reverse order of paths
#     for path in all_paths:
#         path.reverse()
#     optimized_placement = optimize_placement(all_paths, verbose=False)
#     plot_paths(optimized_placement, all_paths, center=True)
#%%
import networkx as nx
# from pyvis.network import Network
import matplotlib.pyplot as plt

graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))
#%%
for mediator in relevant_mediators:
    mediator_gene = protein_id_name[mediator]
    print(f'{mediator_gene:10s}')
    if mediator in wt_mediated_interactions:
        wt_mediated = wt_mediated_interactions[mediator]
    else:
        wt_mediated = []
    if mediator in mut_mediated_interactions:
        mut_mediated = mut_mediated_interactions[mediator]
    else:
        mut_mediated = []

    all_mediated = set(wt_mediated).union(set(mut_mediated))
    all_paths = []
    for src, dst in sorted(all_mediated):
        for path in nx.shortest_paths.all_shortest_paths(graph.to_undirected(), src, dst):
            if mediator in path:
                all_paths.append(path)
    max_len = max([len(path) for path in all_paths])
    # Print all the paths with the mediator in the middle
    mediator_idx = max_len // 2 + 2
    for path in all_paths:
        cols = [f'{"":12s}' for i in range(max_len*2)]
        start_idx = mediator_idx - path.index(mediator)
        for i, node in enumerate(path):
            cols[i+start_idx] = f'{i} {protein_id_name[node]:10s}'
        print(''.join(cols))
        # print(len(path)/2, start_idx)
    print('-'*100)

#%%
for mediator in relevant_mediators:
    mediator_gene = protein_id_name[mediator]
    print(f'{mediator_gene:10s}')
    if mediator in wt_mediated_interactions:
        wt_mediated = wt_mediated_interactions[mediator]
    else:
        wt_mediated = []
    if mediator in mut_mediated_interactions:
        mut_mediated = mut_mediated_interactions[mediator]
    else:
        mut_mediated = []

    all_mediated = set(wt_mediated).union(set(mut_mediated))
    all_paths = []
    for src, dst in sorted(all_mediated):
        for path in nx.shortest_paths.all_shortest_paths(graph.to_undirected(), src, dst):
            if mediator in path:
                all_paths.append(path)
    # Create a graph from the list of paths
    path_graph = nx.DiGraph()
    for path in all_paths:
        for i in range(len(path)-1):
            n1 = protein_id_name[path[i]]
            n2 = protein_id_name[path[i+1]]
            path_graph.add_edge(n1,n2)

    colors = {node: 'grey' for node in path_graph.nodes}
    for path in all_paths:
        src = protein_id_name[path[0]]
        dst = protein_id_name[path[-1]]
        colors[src] = 'green'
        colors[dst] = 'red'
    colors[protein_id_name[mediator]] = 'yellow'

    nx.draw_kamada_kawai(path_graph, with_labels=True, font_size=10, node_color=colors.values())
    plt.show()
# nt = Network('500px', '500px', notebook=True, cdn_resources='in_line')
# nt.from_nx(path_graph)
# nt.show('nx.html')
# %%
