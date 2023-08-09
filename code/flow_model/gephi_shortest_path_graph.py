#%%
import pickle
import networkx as nx

#%%
#%%
wt_tmstp = '20230607_165324'  
mut_tmstp = '20230608_093734'
outdir = f'../../output/'

protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {idx: '/'.join(name) for idx, name in protein_id_name.items()}
# %%
wt_path_graph = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_graph_wildtype.pickle', 'rb'))
mut_path_graph = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_graph_mutant.pickle', 'rb'))
# %%
# Write edges of the shortest path graphs to file
with open(f'{outdir}/{wt_tmstp}/shortest_path_graph_wildtype_edges.csv', 'w') as f:
    f.write('Source\tTarget\tWeight\n')
    for edge in wt_path_graph.edges(data=True):
        f.write(f"{edge[0]}\t{edge[1]}\t{edge[2]['weight']}\n")
with open(f'{outdir}/{mut_tmstp}/shortest_path_graph_mutant_edges.csv', 'w') as f:
    f.write('Source\tTarget\tWeight\n')
    for edge in mut_path_graph.edges(data=True):
        f.write(f"{edge[0]}\t{edge[1]}\t{edge[2]['weight']}\n")
# %%
# Load the pagerank of the nodes in the shortest path graphs
wt_path_pr = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_path_pagerank_wildtype.pickle', 'rb'))
mut_path_pr = pickle.load(open(f'{outdir}/{mut_tmstp}/shortest_path_pagerank_mutant.pickle', 'rb'))
all_nodes = set(wt_path_pr.keys()) & set(mut_path_pr.keys())
wt_path_pr_diff = {node: wt_path_pr[node] - mut_path_pr[node]
                   for node in all_nodes}
mut_path_pr_diff = {node: mut_path_pr[node] - wt_path_pr[node]
                   for node in all_nodes}
#%%
# Write nodes of the shortest path graphs to file
with open(f'{outdir}/{wt_tmstp}/shortest_path_graph_wildtype_nodes.csv', 'w') as f:
    header = 'Id\tLabel\tWT_Pagerank\tMut_Pagerank\tPagerank_Difference'
    f.write(header + '\n')
    for node in all_nodes:
        d = wt_path_pr_diff[node] if node in wt_path_pr_diff else 0
        f.write(f'{node}\t{protein_id_name[node]}\t'\
                f'{wt_path_pr[node]}\t{mut_path_pr[node]}\t{d}\n')
with open(f'{outdir}/{mut_tmstp}/shortest_path_graph_mutant_nodes.csv', 'w') as f:
    header = 'Id\tLabel\tWT_Pagerank\tMut_Pagerank\tPagerank_Difference'
    f.write(header + '\n')
    for node in all_nodes:
        d = mut_path_pr_diff[node] if node in mut_path_pr_diff else 0
        f.write(f'{node}\t{protein_id_name[node]}\t'
                f'{wt_path_pr[node]}\t{mut_path_pr[node]}\t{d}\n')

# %%
# Export the direct connections between the input and output genes
mutant_inputs = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle','rb'))

with open(f'{outdir}/{mut_tmstp}/mut_learned_connections.csv', 'w') as f:
    header = 'Source\tTarget\tWeight\tP_value'
    f.write(header + '\n')

    for target_gene in mutant_inputs:
        for input_gene, p_val in mutant_inputs[target_gene].items():
            if p_val < .01:
                f.write(f'{input_gene}\t{target_gene}\t1.0\t{mutant_inputs[target_gene][input_gene]}\n')
# %%
