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
# ### Hierarchical Clustering

# %%
def refine(state, n=1000):
    s1 = state.entropy()
    for i in range(n): # this should be sufficiently large
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    s2 = state.entropy()
    return state, s1, s2

# %%
hierarchical = gt.inference.minimize_nested_blockmodel_dl(graph)

# %%
r_state_dc, s1, s2 = refine(hierarchical)
print("Hierarchical clustering")
print("Evidence before refinement, after refinement, and difference. Lower is better")
print(s1, s2, s1-s2)
print("Refining again to make sure we are at a local minimum. The difference should be close to zero")
r_state_dc, s1, s2 = refine(hierarchical, 10)
print(s1, s2, s1-s2)

# %%

hierarchical.draw(vertex_text=graph.vertex_properties['name'],
                    vertex_font_size=5,
                    # expand the size of the plot
                    output_size=(2000, 2000)
                    );

# %%
levels = [level for level in hierarchical.levels if level.get_nonempty_B() > 1]

n_vertices = hierarchical.g.num_vertices()
n_levels = len(levels)
vertex_blocks = np.zeros((n_vertices, n_levels), dtype=int)

blocks = hierarchical.levels[0].get_blocks()

for i,block in enumerate(blocks):
    vertex_blocks[i, 0] = block

for i in range(1, n_levels):
    current_blocks = hierarchical.levels[i].get_blocks()
#     print(len(list(current_blocks)))
    for j,last_block in enumerate(vertex_blocks[:,i-1]):
        block = current_blocks[last_block]
        vertex_blocks[j,i] = block

# %%
for i in range(10):
    for j in range(len(levels)):
        print(f'{vertex_blocks[i,j]:4d}',end=',')
    print()

# %% [markdown]
# ## Identify clusters with NANOG

# %%
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle','rb'))
gene_id_to_name = pickle.load(open('../data/gene_id_to_name.pickle','rb'))

names_to_protein_ids = pickle.load(open('../data/protein_names.pickle','rb'))
names_to_gene_ids = pickle.load(open('../data/gene_names.pickle','rb'))

# %%
nanog_ids = names_to_gene_ids['NANOG'] | names_to_protein_ids['NANOG']
nanog_ids

# %%
nanog_vertices = []
for i,vertex in enumerate(graph.vertices()):
#     print(i, vertex)
    for nanog_id in nanog_ids:
        vertex_name = graph.vertex_properties['id'][vertex]
        if vertex_name == nanog_id:
#             assert i==vertex, f'{i}, {vertex}'
            print(i,vertex, vertex_name, vertex_blocks[i])
            nanog_vertices.append(i)

# %% [markdown]
# ## Ranked Block Model

# %%
ranked = NestedBlockState(graph, base_type=RankedBlockState)
for i in range(100):
    ranked.multiflip_mcmc_sweep(beta=np.inf, niter=10)

# %%
r_state_dc, s1, s2 = refine(ranked)
print(s1, s2, s1-s2)

# %%
r_state_dc, s1, s2 = refine(ranked, 200)
print(s1, s2, s1-s2)

# %%
vertex_ranks = []
i=0
for x in r_state_dc.levels[0].get_vertex_order():
    vertex_ranks.append((x,i))
    i+=1
for rank, vertex_num in sorted(vertex_ranks):
    vertex_id = graph.vertex_properties['id'][vertex_num]
    protein_name = protein_id_to_name.get(vertex_id, None)
    print(f'{rank:4d} {vertex_id} {"/".join(protein_name)}')

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


# %% [markdown]
# ## Calculate enrichment of genes in each cluster in the network

# %%
from pprint import pprint
import json
import requests

# %%
go_names = pickle.load(open('../data/go_bio_process_2021_names.pickle','rb'))

# %%
len(go_names)

# %% [markdown]
# # TODO figure out how to get names from blocks

# %%
hierarchical.print_summary()

# %%
levels = hierarchical.get_levels()
for s in levels:
    print(s)
    if s.get_N() == 1:
        break

# %%
from collections import defaultdict

# %%
ids = defaultdict(list)
names = defaultdict(list)
for i,block in enumerate(hierarchical.get_levels()[0].b.a):
    ids[block].append(i)

c=0
for block, node_ids in ids.items():
    for node_id in node_ids:
        ensembl_id = graph.vertex_properties['id'][node_id]
        name_count = 0
        for name in protein_id_to_name[ensembl_id]:
            if name in go_names:
                names[block]+=[name]
                name_count+=1
            else:
                pass
                #print(f'unnamed {ensembl_id}')
        if name_count == 0:
            c+=1
#             print(ensembl_id, name_count)
print("Not in GO", c)
names = dict(names)

# %%
c=0
for id, block in names.items():
    print(len(block))
    c+=len(block)
print('-')
print(c)

# %%
def query_enrichr(names, gene_set_library):
    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/addList'
    genes_str = '\n'.join(names)
    description = 'Example gene list'
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
    }

    response = requests.post(ENRICHR_URL, files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')

    data = json.loads(response.text)
#     print(data)

    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'
    user_list_id = data['userListId']
    response = requests.get(
        ENRICHR_URL + query_string % (user_list_id, gene_set_library)
     )
    if not response.ok:
        raise Exception('Error fetching enrichment results')
    return json.loads(response.content)

# %%
def print_enrichment(response, gene_set_library, threshold=.01, pos_filter_words=None, neg_filter_words=None, n=None):
    if n is None:
        n = len(response[gene_set_library])
    
    for i,enr in enumerate(response[gene_set_library][:n]):
        pvalue = float(enr[6])
        term = enr[1]
        match=False
        if pvalue < threshold:
            if pos_filter_words:
                if any([word in term for word in pos_filter_words]):
                    match=True
            else:
                match=True
            if neg_filter_words:
                if any([word in term for word in neg_filter_words]):
                    match=False
            else:
                match=True
                    
        if match:
            for j in [1, 6]:
                print(headers[j], enr[j])
            print(', '.join(enr[5]))
            print('-')

# %%
gene_set_library = 'GO_Biological_Process_2021'

# %%
headers = ['Rank', 'Term name', 'P-value', 'Z-score', 'Combined score', 'Overlapping genes', 
           'Adjusted p-value', 'Old p-value', 'Old adjusted p-value']

# %%
responses = []
for block_id, block in names.items():
    response = query_enrichr(block, gene_set_library)
    responses.append((block_id, response))
    

# %%
positive_words = ['differentiation', 'development', 'signal', 'matrix', 'organization', 'proliferation', 'stem', 'pathway', 'morpho', 'mesoderm', 'endoderm', 'different', 'specification']
negative_words = ['transcription']

# %%
for block_id, response in responses:
    print("------------------------------------")
    print("BLOCK", block_id)
    print("------------------------------------")    
    print_enrichment(response, gene_set_library, pos_filter_words=positive_words, neg_filter_words=negative_words, n=10)

# %% [markdown]
# ## Export enrichments as pickle

# %%
threshold = .01

enrichments = []
for response in responses:
    for enr in response[1][gene_set_library]:
        pvalue = float(enr[6])
        term = enr[1]
        genes = enr[5]
        enrichments.append((term, pvalue, genes))

pickle.dump(enrichments, open('../data/cluster_enrichments.pickle','wb'))

# %%


# %% [markdown]
# ## Export nodes with cluster assignments

# %%
cluster_file = open('../data/connected_graph_nodes_clusters.csv', 'w')
cluster_file.write('Id,Label,HierarchicalCluster\n')
for block, node_ids in ids.items():
    for node_id in node_ids:
        ensembl_id = graph.vertex_properties['id'][node_id]
        cluster_file.write(','.join([ensembl_id, '/'.join(protein_id_to_name[ensembl_id]), str(block)]))
        cluster_file.write('\n')
cluster_file.close()

# %%



