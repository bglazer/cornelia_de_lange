#%%
import pickle
from itertools import combinations

#%%
enhancer_proteins = pickle.load(open('../data/enhancer_network_proteins.pickle','rb'))

# %%
graph = pickle.load(open('../data/filtered_graph.pickle','rb'))

edges_added = 0
# Add edges between all pairs of proteins that are connected to the same enhancer
for enhancer, proteins in enhancer_proteins.items():
    for protein_a, protein_b in combinations(proteins, r=2):
        if protein_a != protein_b:
            if not graph.has_edge(protein_a, protein_b):
                graph.add_edge(protein_a, protein_b, 
                            directed=False, effect=None, edge_type='shared_enhancer', 
                            evidence=None, weight=None)
                edges_added += 1

            if not graph.has_edge(protein_b, protein_a):
                graph.add_edge(protein_b, protein_a, 
                            directed=False, effect=None, edge_type='shared_enhancer', 
                            evidence=None, weight=None)
                edges_added += 1

print(f'Added {edges_added} edges')
# %%
