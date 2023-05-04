#%%
import scanpy as sc
from matplotlib import pyplot as plt
import numpy as np

#%%
# Set the random seed for reproducibility
np.random.seed(0)
from random import seed
seed(0)

#%%
wt = sc.read_h5ad('../data/wildtype_net.h5ad')
mut = sc.read_h5ad('../data/mutant_net.h5ad')
#%%
wt_ct = wt.obs['cell_type'].value_counts()
mut_ct = mut.obs['cell_type'].value_counts()
# Get a list of all cell types
cell_types = list(set(wt_ct.index).union(set(mut_ct.index)))


# %%
# Randomize the cell type labels and compute the difference in cell type counts
# between the wildtype and mutant to get a null distribution
n_iter = 10_000
rand_diffs = np.zeros((n_iter, len(cell_types)))
wt.obs['condition'] = ['wildtype'] * wt.shape[0]
mut.obs['condition'] = ['mutant'] * mut.shape[0]
combined = wt.concatenate(mut)

for i in range(n_iter):
    # Combine the cells from the wildtype and mutant
    # Randomly shuffle the cell type labels
    combined.obs['cell_type'] = np.random.permutation(combined.obs['cell_type'])
    # Split the combined dataset back into wildtype and mutant
    wt_rand = combined[combined.obs['condition'] == 'wildtype']
    mut_rand = combined[combined.obs['condition'] == 'mutant']
    # Compute the cell type counts
    wt_ct_rand = wt_rand.obs['cell_type'].value_counts()
    mut_ct_rand = mut_rand.obs['cell_type'].value_counts()

    # Compute the difference in cell type counts
    diff = np.abs(mut_ct_rand - wt_ct_rand)
    rand_diffs[i] = diff

# %%
p99 = np.percentile(rand_diffs, 99, axis=0)
diff = np.abs(mut_ct - wt_ct)
#%%
# Make sure the cell types are in the same order
fig, axs = plt.subplots(1, 1, figsize=(15, 5))
wt_ct = wt_ct[cell_types]
mut_ct = mut_ct[cell_types]
# Side by side bar plots
width = .5
axs.bar(np.arange(len(cell_types)), wt_ct, width=width/2, label='Wildtype')
axs.bar(np.arange(len(cell_types))+width/2, mut_ct, width=width/2, label='Mutant')

plt.title('Cell type counts, Wildtype vs Mutant', fontsize=20)
sig_cell_types = [ct + '\n***' if diff[ct] > p99[i] else ct for i, ct in enumerate(cell_types)]
axs.set_xticks(np.arange(len(cell_types))+width/4, sig_cell_types, fontsize=15);
# plt.legend(fontsize=15)
axs.legend()
