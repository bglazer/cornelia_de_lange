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
wt_ct = wt.obs['cell_type'].value_counts()/wt.shape[0]
mut_ct = mut.obs['cell_type'].value_counts()/mut.shape[0]
# Get a list of all cell types
cell_types = list(set(wt_ct.index).union(set(mut_ct.index)))


# %%
# Randomize the cell type labels and compute the difference in cell type counts
# between the wildtype and mutant to get a null distribution
n_iter = 10_000
rand_diffs = np.zeros((n_iter, len(cell_types)))
wt_rows = np.array(wt.obs['cell_type'].values)
mut_rows = np.array(mut.obs['cell_type'].values)
combined = np.concatenate((wt_rows, mut_rows))
cell_types = np.unique(combined)
#%%
for i in range(n_iter):
    # Combine the cells from the wildtype and mutant
    # Randomly shuffle the cell type labels
    shuffled_idxs = np.random.permutation(len(combined))
    # Split the combined dataset back into wildtype and mutant
    wt_rand = combined[shuffled_idxs[0:wt.shape[0]]]
    mut_rand = combined[shuffled_idxs[wt.shape[0]:]]
    # Count the percentage of cells of each type
    wt_rand_ct = np.array([np.sum(wt_rand == ct) for ct in cell_types])/wt_rand.shape[0]
    mut_rand_ct = np.array([np.sum(mut_rand == ct) for ct in cell_types])/mut_rand.shape[0]

    # Compute the difference in cell type percentages
    rand_diffs[i,:] = np.abs(wt_rand_ct - mut_rand_ct)


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
axs.bar(np.arange(len(cell_types)), wt_ct, width=width/2, label='Wildtype', color='#9BA2FF')
axs.bar(np.arange(len(cell_types))+width/2, mut_ct, width=width/2, label='Mutant', color='#EF946C')
# Add an error bar for the 99th percentile of the null distribution
axs.errorbar(np.arange(len(cell_types)), wt_ct, yerr=p99/2, fmt='none', capsize=5, color='black')
axs.errorbar(np.arange(len(cell_types))+width/2, mut_ct, yerr=p99/2, fmt='none', capsize=5, label='99th percentile', color='black')


plt.title('Cell type counts, Wildtype vs Mutant', fontsize=20)
sig_cell_types = [ct + '\n***' if diff[ct] > p99[i] else ct for i, ct in enumerate(cell_types)]
axs.set_xticks(np.arange(len(cell_types))+width/4, sig_cell_types, fontsize=15);
# plt.legend(fontsize=15)
axs.legend()

# %%
