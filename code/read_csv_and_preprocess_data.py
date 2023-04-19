# Description: Convert the csv expression files to h5ad files
#%%
import numpy as np
import scanpy as sc
import pickle
from util import plot_qc_distributions
import scipy

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
def convert_csv(genotype):
    protein_id_to_row = {}
    protein_name_to_id = pickle.load(open('../data/protein_names.pickle', 'rb'))
    
    f = open(f'../data/raw-counts-mes-{genotype}.csv','r')

    # Read the first line of the file
    header = f.readline()

    expression = []
    names_in_data = []
    i=0
    for line in f:
        # Split line by commas
        sp = line.split(',')
        gene = sp[0].strip('"').upper()
        exp = [int(x) for x in sp[1:]]
        expression.append(exp)
        names_in_data.append(gene)
        for protein_id in protein_name_to_id[gene]:
            protein_id_to_row[protein_id] = i
        i+=1
        if i%1000==0:
            print(i, flush=True)

    # Make a sparse matrix from the expression data
    expression = scipy.sparse.csr_matrix(expression)

    adata = sc.AnnData(expression.T, dtype=int)
    adata.var_names = names_in_data
    adata.uns['id_row'] = protein_id_to_row
    return adata

#%%
wt_adata = convert_csv('wildtype')
mut_adata = convert_csv('mutant')
#%%
wt_adata.uns['genotype'] = 'wildtype'
mut_adata.uns['genotype'] = 'mutant'
#%%
# Make the raw counts their own layer
wt_adata.layers['raw'] = wt_adata.X.copy()
mut_adata.layers['raw'] = mut_adata.X.copy()

#%%
def add_metadata(adata):
    genotype = adata.uns['genotype']
    f = open(f'../data/metadata-mes-{genotype}.csv')
    header = next(f)

    cell_lines = {}
    cell_types = {}
    for line in f:
        tag, umap1, umap2, cell_line, cell_type = line.strip().split(',')
        tag = tag.strip('"')
        cell_line = cell_line.strip('"')
        cell_type = cell_type.strip('"')
        cell_lines[tag] = cell_line
        cell_types[tag] = cell_type

    # Get header with cell names from the expression file
    f = open(f'../data/raw-counts-mes-{genotype}.csv')
    header = next(f)
    header = header.strip().split(',')
    cell_tags = [tag.strip('"') for tag in header[1:]]

    # Get cells in the same order as the expression file
    cell_lines_ordered = []
    cell_types_ordered = []
    for tag in cell_tags:
        cell_lines_ordered.append(cell_lines[tag])
        cell_types_ordered.append(cell_types[tag])

    # Add cell metadata to the AnnData objects
    adata.obs['cell_type'] = cell_types_ordered
    adata.obs['cell_line'] = cell_lines_ordered
    return adata

#%%
wt_adata = add_metadata(wt_adata)
mut_adata = add_metadata(mut_adata)

#%%
def plot_qc(adata):
    genotype = adata.uns['genotype']
    sc.pl.highest_expr_genes(adata, n_top=20)

    # Filter out genes that are not expressed in at least 3 cells
    sc.pp.filter_genes(adata, min_cells=3)

    # Filter out cells that have less than 200 genes expressed
    sc.pp.filter_cells(adata, min_genes=200)

    # Plot the percentage of mitochondrial genes expressed per cell
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    plot_qc_distributions(adata, genotype, 'all_genes', '../figures')

# plot_qc(wt_adata)
# plot_qc(mut_adata)

#%%
def preprocess(adata):
    # Add another layer for the preprocessed data
    adata.layers['preprocessed'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer='preprocessed')
    sc.pp.log1p(adata, layer='preprocessed')
    # Make the main X matrix the preprocessed data
    adata.X = adata.layers['preprocessed']
    # The regress_out and scale operations gave me results that were difficult to interpret
    # (i.e. negative values for the gene expression). They're not universally 
    # recommended for pseudotime analysis, so I'm skipping them for now
    # sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    # sc.pp.scale(adata, max_value=10)
    return adata

wt_adata = preprocess(wt_adata)
mut_adata = preprocess(mut_adata)

# %%
# Prune to just the genes in the network
from util import filter_to_network

#%%
wt_net = filter_to_network(wt_adata)
mut_net = filter_to_network(mut_adata)

#%%
# Save the network filtered AnnData objects as h5ad files
wt_net.write('../data/wildtype_net.h5ad')
mut_net.write('../data/mutant_net.h5ad')
# %%
# Save the unfiltered AnnData objects as h5ad files
wt_adata.write('../data/wildtype_full.h5ad')
mut_adata.write('../data/mutant_full.h5ad')
# %%
