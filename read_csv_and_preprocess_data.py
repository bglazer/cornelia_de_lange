# Description: Convert the csv expression files to h5ad files
#%%
import numpy as np
import scanpy as sc
import pickle

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

    expression = np.array(expression)

    adata = sc.AnnData(expression.T)
    adata.var_names = names_in_data

    return adata, protein_id_to_row

#%%
wt_adata, wt_id_row = convert_csv('wildtype')
mut_adata, mut_id_row = convert_csv('mutant')
#%%
pickle.dump(wt_id_row, open('../data/wildtype_id_row.pickle', 'wb'))
pickle.dump(mut_id_row, open('../data/mutant_id_row.pickle', 'wb'))

#%%
# Load file
for genotype in ['wildtype', 'mutant']:
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
    if genotype == 'wildtype':
        wt_adata.obs['cell_type'] = cell_types_ordered
        wt_adata.obs['cell_line'] = cell_lines_ordered
    else:
        mut_adata.obs['cell_type'] = cell_types_ordered
        mut_adata.obs['cell_line'] = cell_lines_ordered

# %%
# Save the AnnData objects as h5ad files
wt_adata.write('../data/wildtype.h5ad')
mut_adata.write('../data/mutant.h5ad')
# %%
