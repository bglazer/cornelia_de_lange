# Description: Convert the csv expression files to h5ad files
#%%
import numpy as np
import scanpy as sc

#%%
# Set the random seed for reproducibility
import numpy as np
np.random.seed(42)
from random import seed
seed(42)

#%%
def convert_csv(genotype):
    f = open(f'../data/raw-counts-mes-{genotype}.csv','r')

    # Read the first line of the file
    header = f.readline()

    expression = []
    names_in_data = []
    for line in f:
        # Split line by commas
        sp = line.split(',')
        gene = sp[0]
        exp = [int(x) for x in sp[1:]]
        expression.append(exp)
        names_in_data.append(gene.strip('"').upper())

    expression = np.array(expression)

    adata = sc.AnnData(expression.T)
    adata.var_names = names_in_data

    return adata

#%%
wt_adata = convert_csv('wildtype')
mut_adata = convert_csv('mutant')
# %%
# Save the AnnData objects as h5ad files
wt_adata.write('../data/wildtype.h5ad')
mut_adata.write('../data/mutant.h5ad')
# %%
