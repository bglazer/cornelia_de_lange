# %% [markdown]
# # Bayesian Graph Ranking
# Order nodes using a bayesian block model with a hierarchical, ranked structure
# This gives a ranking of the nodes in the graph based on the structure of edges in the graph

# %%
import pickle
import networkx as nx
from graph_tool.all import *
import numpy as np
import graph_tool as gt
from nx2gt import nx2gt

# %%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
# Set the seed in graphtool
gt.seed_rng(42)
# Set the seed in python
import random
random.seed(42)

# %%
graph = pickle.load(open('../data/filtered_graph.pickle','rb'))

# %%
# Convert to graphtool graph from networkx
graph = nx2gt(graph)

# %%
print("Number of edges: ", graph.num_edges())
print("Number of nodes: ", graph.num_vertices())

# %%
gene_id_to_name = pickle.load(open('../data/gene_id_to_name.pickle','rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle','rb'))

# %%
# Add gene name as a vertex property
graph.vertex_properties['name'] = graph.new_vertex_property('string')
for gene_index in graph.vertices():
    gene_id = graph.vertex_properties['id'][gene_index]
    gene_name = '/'.join(protein_id_to_name.get(gene_id, None))
    # Add the gene name as a vertex property
    graph.vertex_properties['name'][gene_index] = gene_name

# %% [markdown]

# %%
def refine(state, n=1000):
    s1 = state.entropy()
    for i in range(n): # this should be sufficiently large
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    s2 = state.entropy()
    return state, s1, s2

# %% [markdown]
# ## Ranked Block Model

# %%
ranked = NestedBlockState(graph, base_type=RankedBlockState)
ranked.multiflip_mcmc_sweep(beta=np.inf, niter=20000)

# %%
r_state_dc, s1, s2 = refine(ranked)
print("Hierarchical clustering")
print("Evidence before refinement, after refinement, and difference. Lower is better")
print(s1, s2, s1-s2)
print("Refining again to make sure we are at a local minimum. The difference should be close to zero")
r_state_dc, s1, s2 = refine(ranked, 10)
print(s1, s2, s1-s2)

# %%
vertex_ranks = []
ranking = []
i=0
for x in r_state_dc.levels[0].get_vertex_order():
    vertex_ranks.append((x,i))
    i+=1
for rank, vertex_num in sorted(vertex_ranks):
    vertex_id = graph.vertex_properties['id'][vertex_num]
    protein_name = protein_id_to_name.get(vertex_id, None)
    print(f'{rank:4d} {vertex_id} {"/".join(protein_name)}')
    ranking.append((vertex_id, protein_name))
pickle.dump(ranking, open('../data/graph_ranked_genes.pickle','wb'))
# %%
# Draw the ranked hierarchical block model
pos = sfdp_layout(graph, cooling_step=0.99, multilevel=False, R=50000,
                    rmap=r_state_dc.levels[0].get_vertex_order(),
                    )

# # Stretch the layout somewhat
# for v in graph.vertices():
#     pos[v][1] *= 40
#     pos[v][0] *= 20

ranked.levels[0].draw(pos=pos)

# %%
# Output the ranking as a csv file for Gephi

with open('../data/ranked_genes.csv', 'w') as f:
    f.write('Id,Rank\n')
    for rank, (gene_id, gene_name) in enumerate(ranking):
        f.write(f'{gene_id},{rank}\n')

# %%
