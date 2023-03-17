#%%
from matplotlib import pyplot as plt
import numpy as np

#%%
f = open('../data/raw-counts-mes-wildtype.csv','r')

#%%
# Read the first line of the file
header = f.readline()

# %%
expression = []
names_in_data = []
for line in f:
    # Split line by commas
    sp = line.split(',')
    gene = sp[0]
    exp = [int(x) for x in sp[1:]]
    expression.append(exp)
    names_in_data.append(gene.strip('"'))
expression = np.array(expression)

#%%
# Distribution of total of each gene's expression across all cells
plt.hist(expression.sum(axis=1), bins=100);
# %%
# Distribution of total of all gene expression in each cells
plt.hist(expression.sum(axis=0), bins=100);

# %%
import umap 
# import PCA
from sklearn.decomposition import PCA

#%%
# First project down to 100 dimensions using PCA
pca = PCA(n_components=100)
pcs = pca.fit_transform(expression.T)

# %%
two_d = umap.UMAP().fit_transform(pcs)
# %%
# Plot the UMAP embedding
plt.scatter(two_d[:,0], two_d[:,1], s=1, alpha=0.1);

############
# TODO ###
# 1. Batch correct the data
# 2. Zero-one binarize the data using a more sophisticated method
############

# %%
# Binarize the data
# Median of each gene's expression across all cells is the threshold
# Must be strictly greater than the median. Since the data is zero skewed
# this means many genes have more than 50% zeros in the binarized data. 
# In fact, we get a variety of distributions of the binarized data.
medians = np.median(expression, axis=1).astype(int).reshape(-1,1)
binary_expression = (expression > medians)
# Save the binary data
import pickle
pickle.dump(binary_expression, open('../data/binary_expression.pickle', 'wb'))

#%%
# Distribution of the proportion of genes that each cell expresses
plt.hist(binary_expression.mean(axis=0),bins=100)

#%%
# Distribution of the proportion of cells that each gene is expressed in
plt.hist(binary_expression.mean(axis=1),bins=100)

# %%
# PCA embedding of the binary data
# binary_pcs = PCA(n_components=100).fit_transform(binary_expression)
# Umap embedding of the binary data
binary_embedding = umap.UMAP(metric='manhattan').fit_transform(binary_expression.T)
# Plot the UMAP embedding
plt.scatter(binary_embedding[:,0], binary_embedding[:,1], s=1, alpha=0.1);

# %%
# Import gene network from Tiana et al paper
graph = pickle.load(open('../data/filtered_graph.pickle', 'rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle', 'rb'))
protein_name_to_ids = pickle.load(open('../data/protein_names.pickle', 'rb'))

# %%
# Get the indexes of the data rows that correspond to nodes in the Nanog regulatory network
indices_of_nodes_in_graph = []
for i,name in enumerate(names_in_data):
    name = name.upper()
    if name in protein_name_to_ids:
        for id in protein_name_to_ids[name]:
            if id in graph.nodes:
                indices_of_nodes_in_graph.append(i)

# %%
# UMAP embed only the genes that are in the Nanog regulatory network
graph_embedding = umap.UMAP(metric='manhattan').fit_transform()
# %%
# Plot the UMAP embedding
plt.scatter(graph_embedding[:,0], graph_embedding[:,1], s=1, alpha=0.1);

#%%
!mkdir ../figures/histograms_of_genes_in_graph/ -p

#%%
# Sort the data indices of the genes in the network by their expression
network_expression = expression[indices_of_nodes_in_graph,:]
sorted_expression_idxs = network_expression.sum(axis=1).argsort()
sorted_data_indices = np.array(indices_of_nodes_in_graph)[sorted_expression_idxs]

#%%
# Plot the histograms of the expression of the genes in the Nanog regulatory network
for i,data_idx in enumerate(sorted_data_indices[::-1][:10]):
    if i%10==0:
        print(i)
    plt.hist(expression[data_idx,:], bins=100)
    # Vertical red line on at the median
    plt.axvline(np.median(expression[data_idx,:]), color='red', linewidth=2)
    plt.title(names_in_data[data_idx])
    plt.savefig(f'../figures/histograms_of_genes_in_graph/{i}_{names_in_data[data_idx]}.png')
    # Clear the plot so we can plot the next histogram
    plt.clf();

#$# %%

# %%
