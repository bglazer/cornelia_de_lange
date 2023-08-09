#%%
import pickle
import scanpy as sc
from collections import Counter
import matplotlib.pyplot as plt

#%%
wt_tmstp = '20230607_165324'  
mut_tmstp = '20230608_093734'
outdir = f'../../output/'
#%%
adata = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mutant_inputs = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle','rb'))
wildtype_inputs = pickle.load(open(f'{outdir}/{wt_tmstp}/input_selection_pvalues_wildtype.pickle','rb'))
#%%
wt_only = dict()
mut_only = dict()
wt_counts = Counter()
mut_counts = Counter()
for target_idx in wildtype_inputs:
    wt_selected = set([idx for idx, pval in wildtype_inputs[target_idx].items() if pval < .01])
    mut_selected = set([idx for idx, pval in mutant_inputs[target_idx].items() if pval < .01])
    in_wt_not_in_mut = wt_selected - mut_selected
    in_mut_not_in_wt = mut_selected - wt_selected
    
    for src_idx in in_wt_not_in_mut:
        if src_idx not in wt_only:
            wt_only[src_idx] = []
        wt_only[src_idx].append(target_idx)
    for src_idx in in_mut_not_in_wt:
        if src_idx not in mut_only:
            mut_only[src_idx] = []
        mut_only[src_idx].append(target_idx)

    wt_counts.update(wt_selected)
    mut_counts.update(mut_selected)

    if len(in_wt_not_in_mut) > 0 or len(in_mut_not_in_wt) > 0:
        print(f'For target {adata.var_names[target_idx]}:')
        if len(in_wt_not_in_mut) > 0:
            print('In wildtype but not in mutant:')
            for idx in in_wt_not_in_mut:
                print(f'{adata.var_names[idx]}')
        if len(in_mut_not_in_wt) > 0:    
            print('In mutant but not in wildtype:')
            for idx in in_mut_not_in_wt:
                print(f'{adata.var_names[idx]}')
        print('-'*40)
              

# %%
print('Genes that are most frequently in wildtype but not mutant')
for idx,targets in sorted(wt_only.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f'{adata.var_names[idx]:8s}: {len(targets):3d} {wt_counts[idx]}')

# # %%
print('Genes that are most frequently in mutant but not wildtype')
for idx,targets in sorted(mut_only.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f'{adata.var_names[idx]:8s}: {len(targets):3d} {mut_counts[idx]}')
# %%
for src_idx in wt_only:
    print(f'Genes targeted by {adata.var_names[src_idx]} in wildtype but not mutant')
    for target_idx in wt_only[src_idx]:
        print(f'{adata.var_names[target_idx]}')
#%%
for src_idx in mut_only:
    print(f'Genes targeted by {adata.var_names[src_idx]} in mutant but not wildtype')
    for target_idx in mut_only[src_idx]:
        print(f'{adata.var_names[target_idx]}')
# print('Genes targeted by gene in both wildtype and mutant')
# for idx in common_targets:
#     print(f'{adata.var_names[idx]}')
# %%
import networkx as nx
# %%
wt_graph = nx.DiGraph()
mut_graph = nx.DiGraph()
for target_idx in wildtype_inputs:
    wt_selected = set([idx for idx, pval in wildtype_inputs[target_idx].items() if pval < .01])
    mut_selected = set([idx for idx, pval in mutant_inputs[target_idx].items() if pval < .01])
    wt_graph.add_edges_from([(idx, target_idx) for idx in wt_selected])
    mut_graph.add_edges_from([(idx, target_idx) for idx in mut_selected])
    
# %%
# Compute the strongly connected components
wt_scc = list(nx.strongly_connected_components(wt_graph))
mut_scc = list(nx.strongly_connected_components(mut_graph))
print(f'Number of strongly connected components in wildtype: {len(wt_scc)}')
print(f'Number of strongly connected components in mutant: {len(mut_scc)}')

# %%
# Compute the undirected connected components
wt_cc = list(nx.connected_components(wt_graph.to_undirected()))
mut_cc = list(nx.connected_components(mut_graph.to_undirected()))
print(f'Number of connected components in wildtype: {len(wt_cc)}')
print(f'Number of connected components in mutant: {len(mut_cc)}')
# %%
# Compute the betweenness centrality of the neural network input graph
wt_bc = nx.betweenness_centrality(wt_graph)
mut_bc = nx.betweenness_centrality(mut_graph)
# Find the nodes with the highest betweenness centrality
wt_bc_top = sorted(wt_bc.items(), key=lambda x: x[1], reverse=True)[:20]
mut_bc_top = sorted(mut_bc.items(), key=lambda x: x[1], reverse=True)[:20]

print('Nodes with the highest betweenness centrality in wildtype')
for idx, bc in wt_bc_top:
    print(f'{adata.var_names[idx]:8s}: {bc:.3f}')
print('Nodes with the highest betweenness centrality in mutant')
for idx, bc in mut_bc_top:
    print(f'{adata.var_names[idx]:8s}: {bc:.3f}')
print('Nodes with biggest difference in betweenness centrality (WT - Mut)')
diffs = [(idx, wt_bc[idx] - mut_bc[idx]) for idx in wt_bc]
sorted_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
for idx, diff_bc in sorted_diffs[:20]:
    print(f'{adata.var_names[idx]:8s}: {diff_bc:.3f}')
for idx, diff_bc in sorted_diffs[::-1][:20]:
    print(f'{adata.var_names[idx]:8s}: {diff_bc:.3f}')
#%%
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {idx: '/'.join(name) for idx, name in protein_id_name.items()}
#%%
         
# %%
# Biggest changes in path node counts
wt_path_node_counts = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_table_wildtype.pickle', 'rb'))
mut_path_node_counts = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_table_mutant.pickle', 'rb'))
# %%
# Which nodes have the biggest change in the number of targets 
# for which they are over-represented in the shortest paths?
diffs = []
for idx in set(wt_path_node_counts.keys()) | set(mut_path_node_counts.keys()):
    wt_count = wt_path_node_counts[idx]
    mut_count = mut_path_node_counts[idx]
    denom = max(wt_count, mut_count)
    diff = wt_count - mut_count
    if denom == 0:
        pct = 0
    else:
        pct = diff / denom
    diffs.append((idx, diff, pct, wt_count, mut_count))

sorted_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
pickle.dump(sorted_diffs, open(f'{outdir}/mediator_path_node_count_diffs.pickle', 'wb'))
#%%
print('More frequently a mediator in wildtype')
for idx, diff, pct, wt_count, mut_count in sorted_diffs[:20]:
    print(f'{protein_id_name[idx]:8s}: {diff:3d} {pct:.3f} {wt_count} {mut_count}')
#%%
print('More frequently a mediator in mutant')
for idx, diff, pct, wt_count, mut_count in sorted_diffs[::-1][:20]:
    print(f'{protein_id_name[idx]:8s}: {abs(diff):3d} {abs(pct):.3f} {wt_count} {mut_count}')

# %%
wt_overrepresented_node_paths = pickle.load(open(f'{outdir}/{wt_tmstp}/overrepresented_node_paths_wildtype.pickle', 'rb'))
mut_overrepresented_node_paths = pickle.load(open(f'{outdir}/{mut_tmstp}/overrepresented_node_paths_mutant.pickle', 'rb'))
# %%
# What are the targets of the most changed nodes?
wt_mediator_targets = {}
for node in wt_overrepresented_node_paths:
    wt_mediator_targets[node] = set()
    # Find the set of paths where the mediator is overrepresented
    # then add the first and last node of each path to the set of targets
    for path in wt_overrepresented_node_paths[node]:
        src = path[0]
        tgt = path[-1]
        wt_mediator_targets[node].add(src)
        wt_mediator_targets[node].add(tgt)
mut_mediator_targets = {}
for node in mut_overrepresented_node_paths:
    mut_mediator_targets[node] = set()
    for path in mut_overrepresented_node_paths[node]:
        src = path[0]
        tgt = path[-1]
        mut_mediator_targets[node].add(src)
        mut_mediator_targets[node].add(tgt)
wt_only_mediator_targets = {}
mut_only_mediator_targets = {}
for node in (wt_mediator_targets.keys() | mut_mediator_targets.keys()):
    if node not in mut_mediator_targets:
        wt_only_mediator_targets[node] = wt_mediator_targets[node]
        continue
    if node not in wt_mediator_targets:
        mut_only_mediator_targets[node] = mut_mediator_targets[node]
        continue
    wt_not_mut = wt_mediator_targets[node] - mut_mediator_targets[node]
    wt_only_mediator_targets[node] = wt_not_mut
    mut_not_wt = mut_mediator_targets[node] - wt_mediator_targets[node]
    mut_only_mediator_targets[node] = mut_not_wt

pickle.dump(wt_mediator_targets, open(f'{outdir}/{wt_tmstp}/mediator_targets_wildtype.pickle', 'wb'))
pickle.dump(mut_mediator_targets, open(f'{outdir}/{mut_tmstp}/mediator_targets_mutant.pickle', 'wb'))
pickle.dump(wt_only_mediator_targets, open(f'{outdir}/{wt_tmstp}/mediator_targets_wildtype_only.pickle', 'wb'))
pickle.dump(mut_only_mediator_targets, open(f'{outdir}/{mut_tmstp}/mediator_targets_mutant_only.pickle', 'wb'))
# %%
print('Number of genes connected to mediators in shortest paths in wildtype')
for protein_id, *_ in sorted_diffs[:20]:
    print(f'{protein_id_name[protein_id]:8s} {len(wt_only_mediator_targets[protein_id]):3d}')
# %%
print('Number of genes connected to mediators in shortest paths in mutant')
for protein_id, *_ in sorted_diffs[::-1][:20]:
    print(f'{protein_id_name[protein_id]:8s} {len(mut_only_mediator_targets[protein_id]):3d}')

# %%
for protein_id, *_ in sorted_diffs[::-1][:20]:
    print('-------')
    print(f'{protein_id_name[protein_id]:8s}')
    print('-------')
    for tgt in mut_only_mediator_targets[protein_id]:
        print(f'{protein_id_name[tgt]:8s}')
