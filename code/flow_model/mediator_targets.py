#%%
import pickle
import networkx as nx
import scanpy as sc
#%%
mut_tmstp = '20230608_093734'

genotype = 'mutant'
outdir = f'../../output/{mut_tmstp}'
mut = sc.read_h5ad(f'../../data/mutant_net.h5ad')
#%%
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {idx: '/'.join(name) for idx, name in protein_id_name.items()}

#%%
# Load the genes that are significantly different in pagerank
# between the wildtype and mutant
diff_pagerank_file = open(f'{outdir}/pagerank_diff_significant_mutant.pickle', 'rb')
diff_pagerank_genes = pickle.load(diff_pagerank_file)
diff_pagerank_file.close()

#%% 
# Load the genes that are significantly different in degree centrality
# between the wildtype and mutant
diff_degree_centrality_file = open(f'{outdir}/centrality_diff_mut.pickle', 'rb')
diff_degree_centrality_genes = pickle.load(diff_degree_centrality_file)
diff_degree_centrality_file.close()

#%%
# Load the shortest paths
shortest_paths = pickle.load(open(f'{outdir}/shortest_paths_{genotype}.pickle', 'rb'))
random_shortest_paths = pickle.load(open(f'{outdir}/random_shortest_paths_{genotype}.pickle', 'rb'))
#%%
# Combine pagerank and degree centrality genes
mediators = set(diff_pagerank_genes) | set(diff_degree_centrality_genes)

#%%
sig_path_nodes = pickle.load(open(f'{outdir}/sig_path_nodes_{genotype}.pickle', 'rb'))

#%% 
# For each mediator, find the targets for which it appears in the shortest paths more often
# than expected by chance
mediator_targets = {}
for mediator in mediators:
    mediator_targets[mediator] = []
    for target, sig_nodes in sig_path_nodes.items():
        if mediator in sig_nodes:
            mediator_targets[mediator].append(target)

#%%
# Save the mediator targets
pickle.dump(mediator_targets, open(f'{outdir}/mediator_targets_{genotype}.pickle', 'wb'))

# %%
