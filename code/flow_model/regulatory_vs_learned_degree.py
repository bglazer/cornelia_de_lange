#%%
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%%
# Load the input selection p-values
mut_tmstp = '20230608_093734'
outdir = f'../../output/'
mutant_inputs = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle','rb'))
#%%
learned_graph = nx.DiGraph()
for target_gene in mutant_inputs:
    for input_gene, p_val in mutant_inputs[target_gene].items():
        if p_val < .01:
            learned_graph.add_edge(input_gene, target_gene)
#%%
# Load the regulatory network
regulatory_graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))
# %%
all_nodes = set(learned_graph.nodes) & set(regulatory_graph.nodes)

#%%
# For node in all nodes, get the in-degree and out-degree in the learned graph and the regulatory graph
learned_degree = {node: learned_graph.degree(node) for node in all_nodes}
regulatory_degree = {node: regulatory_graph.degree(node) for node in all_nodes}
# %%
degrees = []
for node in all_nodes:
    degrees.append((learned_degree[node], regulatory_degree[node]))
degrees = np.array(degrees)
#%%
plt.scatter(degrees[:,0], degrees[:,1], s=3)
plt.xlabel('Learned Degree')
plt.ylabel('Regulatory Degree')
plt.title('Degree in Learned vs Regulatory Graph')

# %%
# Compute the betweenness centrality of each node in the learned graph and the regulatory graph
learned_betweenness = nx.betweenness_centrality(learned_graph)
regulatory_betweenness = nx.betweenness_centrality(regulatory_graph)
#%%
betweenness = []
for node in all_nodes:
    betweenness.append((learned_betweenness[node], regulatory_betweenness[node]))
betweenness = np.array(betweenness)
#%%
plt.scatter(betweenness[:,0], betweenness[:,1], s=3)
plt.xlabel('Learned Betweenness Centrality')
plt.ylabel('Regulatory Betweenness Centrality')
plt.title('Betweenness Centrality in Learned vs Regulatory Graph')

# %%
