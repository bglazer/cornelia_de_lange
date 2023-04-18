#%%
import scanpy as sc
from util import filter_to_network

#%%
mut_velo = sc.read_h5ad('../data/scvelo-mesoderm-12-mutant.h5ad')
# %%
wt_velo = sc.read_h5ad('../data/scvelo-mesoderm-12-wildtype.h5ad')
# %%
# Filter to just the genes that are in the graph
mut_velo_net, mut_id_row, mut_id_new_row = filter_to_network(mut_velo)
wt_velo_net, wt_id_row, wt_id_new_row = filter_to_network(wt_velo)
# Make the gene names uppercase
mut_velo_net.var_names = [name.upper() for name in mut_velo_net.var_names]
wt_velo_net.var_names = [name.upper() for name in wt_velo_net.var_names]

# %%
# Save the filtered data
mut_velo_net.write('../data/mutant_velocity_network.h5ad')
wt_velo_net.write('../data/wildtype_velocity_network.h5ad')
# %%
