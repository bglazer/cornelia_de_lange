# Archetype analysis
#%%
# Jupyter magic to reload modules
# Ignore pylance warnings
%load_ext autoreload
%autoreload 2
#%%
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polytope


#%%
dataset = 'net'
wt = sc.read_h5ad(f'../data/wildtype_{dataset}.h5ad')
mut = sc.read_h5ad(f'../data/mutant_{dataset}.h5ad')

# %%
# Find the number of archetypes in each dataset
wt_n_archetypes = wt.obsm['arc_distance'].shape[1]
mut_n_archetypes = mut.obsm['arc_distance'].shape[1]
print(f'Wildtype has {wt_n_archetypes} archetypes')
print(f'Mutant has {mut_n_archetypes} archetypes')

#%%
n_pcs = 3
n_archetypes = max(wt_n_archetypes, mut_n_archetypes)
fig, axs = plt.subplots(2*n_pcs, n_archetypes+1, figsize=(10,5*n_pcs))

wt_umap = wt.obsm['X_umap']
mut_umap = mut.obsm['X_umap']
wt_pca = wt.obsm['X_pca']
mut_pca = mut.obsm['X_pca']

# Get consistent set of color codes for both mutant and wildtype
cell_types = pd.concat([wt.obs['cell_type'], mut.obs['cell_type']]).unique()
color_codes = {c: i for i,c in enumerate(cell_types)}
wt_cell_colors = [color_codes[c] for c in wt.obs['cell_type']]
mut_cell_colors = [color_codes[c] for c in mut.obs['cell_type']]

for j in range(0, n_pcs):
    row = j*2
    pc1 = row
    pc2 = row+1

    axs[row][0].scatter(wt_pca[:,pc1], wt_pca[:,pc2], c=wt_cell_colors, cmap = 'tab20', s=.1)
    axs[row+1][0].scatter(mut_pca[:,pc1], mut_pca[:,pc2], c=mut_cell_colors, cmap = 'tab20', s=.1)
    axs[row][0].set_title('Wildtype')
    axs[row+1][0].set_title('Mutant')
    for i in [0,1]:
        axs[row+i][0].set_xticks([])
        axs[row+i][0].set_yticks([])
        axs[row+i][0].set_xlabel(f'PC{pc1}')
        axs[row+i][0].set_ylabel(f'PC{pc2}')
        
    def plot_archetype_dist(data, ax, i):
        archetype_score = data.obs[f'Arc_{i}_PCHA_Score']
        pca = data.obsm['X_pca']
        ax.scatter(pca[:,pc1], pca[:,pc2], c = archetype_score, cmap = 'RdBu', s=.2)
        ax.set_title(f'Archetype {i}')
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    # Color cells by their distance to each archetype
    for i in range(0,n_archetypes):
        col = i+1
        if i < wt_n_archetypes:
            axs[row][col] = plot_archetype_dist(wt, axs[row,col], i+1)
        else:
            # If there are more archetypes in the mutant, plot a blank subplot
            axs[row][col].axis('off')
        if i < mut_n_archetypes:
            axs[row+1][col] = plot_archetype_dist(mut, axs[row+1,col], i+1)
        else:
            axs[row+1][col].axis('off')

    plt.tight_layout()


# %%
# Find the percentage of each cell type in each archetype's "specialists"
cell_type_pcts = {'wt':{}, 'mut':{}}

for i in range(1,wt_n_archetypes+1):
    specialists = wt.obs['specialists_pca_diffdist'] == f'Arc_{i}'
    pcts = wt.obs['cell_type'][specialists].value_counts(normalize=True)
    cell_type_pcts['wt'][f'Arc_{i}'] = pcts

for i in range(1,mut_n_archetypes+1):
    specialists = mut.obs['specialists_pca_diffdist'] == f'Arc_{i}'
    pcts = mut.obs['cell_type'][specialists].value_counts(normalize=True)
    cell_type_pcts['mut'][f'Arc_{i}'] = pcts

# %%
# Plot the percentage of each cell type in each archetype's "specialists" as a horizontal stacked bar chart
# Each bar represents an archetype, split between the cell types that make up the archetype
def percentage_chart(results, ax, add_colname=True):
    labels = list(results.index)
    category_names = list(results.columns)
    data = results.values

    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['tab20'](
        np.linspace(0.15, 0.85, data.shape[1]))

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, 1)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        # Format the labels to be percentages
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        xcenters = starts + widths / 2
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            # Only add a label if the percentage is greater than 5%
            if c > 0.05:
                # Label with the percentage and the cell type
                bar_label = f'{c:.0%} {colname}' if add_colname else f'{c:.0%}'
                ax.text(x, y, bar_label, ha='center', va='center',
                        color=text_color)
        # ax.bar_label(rects, label_type='center', color=text_color)
    # Put a legend in the upper right corner, outside the plot
    return ax

fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0] = percentage_chart(pd.DataFrame(cell_type_pcts['wt']).T, axs[0])
axs[1] = percentage_chart(pd.DataFrame(cell_type_pcts['mut']).T, axs[1])
axs[1].legend(ncol=1, bbox_to_anchor=(1, 1),
              loc='upper left', fontsize='small')
axs[0].set_title('Wildtype')
axs[1].set_title('Mutant')

plt.tight_layout()

# %%
# Get the percentage of generalists in each dataset
# Generalists are annotated with NaN
generalists = {'wt':{}, 'mut':{}}
generalists['wt'] = wt.obs['specialists_pca_diffdist'].isna()
generalists['mut'] = mut.obs['specialists_pca_diffdist'].isna()
print("Generalist counts")
print(f'Wildtype - {generalists["wt"].sum()}')
print(f'Mutant   - {generalists["mut"].sum()}')
print("Percentage Generalists")
print(f'Wildtype - {generalists["wt"].sum()/len(generalists["wt"]):.2%}')
print(f'Mutant   - {generalists["mut"].sum()/len(generalists["mut"]):.2%}')

# %%
# Plot the percentage of generalists (i.e. not specialists) that are each cell type 
generalists_cell_type = {'wt':{}, 'mut':{}}
generalists_cell_type['wt'] = wt.obs['cell_type'][generalists['wt']].value_counts(normalize=True)
generalists_cell_type['mut'] = mut.obs['cell_type'][generalists['mut']].value_counts(normalize=True)
# %%
generalist_cell_type_df = pd.DataFrame(generalists_cell_type).T
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax = percentage_chart(generalist_cell_type_df, ax, add_colname=False)
ax.set_title('Generalists')
ax.legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')
plt.tight_layout()
print(generalist_cell_type_df.T)

#%%
if 'cluster_sums' not in wt.obsm.keys():
    exit()

# %%
# Make a heatmap of archetype  versus cluster sum
# Fill NaNs with 0
wt.obs['specialists_pca_diffdist'].cat.add_categories('Generalist', inplace=True)
wt.obs['specialists_pca_diffdist'] = wt.obs['specialists_pca_diffdist'].fillna('Generalist')
# Reset the index of the specialist column so that it can be used as a column
wt.obs['specialists_pca_diffdist'].reset_index(drop=True, inplace=True)
#%%
mut.obs['specialists_pca_diffdist'].cat.add_categories('Generalist', inplace=True)
mut.obs['specialists_pca_diffdist'] = mut.obs['specialists_pca_diffdist'].fillna('Generalist')
mut.obs['specialists_pca_diffdist'].reset_index(drop=True, inplace=True)


#%%
wt_cluster_sum = pd.DataFrame(wt.obsm['cluster_sums']).groupby(wt.obs['specialists_pca_diffdist']).sum()
mut_cluster_sum = pd.DataFrame(mut.obsm['cluster_sums']).groupby(mut.obs['specialists_pca_diffdist']).sum()

# Normalize the sum of each cluster to 1
wt_cluster_mean = wt_cluster_sum.div(wt_cluster_sum.sum(axis=0), axis=1)
mut_cluster_mean = mut_cluster_sum.div(mut_cluster_sum.sum(axis=0), axis=1)

# Compute the overrepresentation of each cluster in each archetype
# What percentage of cells are in each archetype?
wt_archetype_pct = wt.obs['specialists_pca_diffdist'].value_counts(normalize=True)
mut_archetype_pct = mut.obs['specialists_pca_diffdist'].value_counts(normalize=True)
# For each archetype, compute the difference between the percentage of cells in that archetype
#  and the percentage of cluster expression in the archetype
wt_pct_diffs = wt_cluster_mean.sort_index().values - wt_archetype_pct.sort_index().values[:,None]
wt_pct_diffs = pd.DataFrame(wt_pct_diffs, index=wt_cluster_mean.index, columns=wt_cluster_mean.columns)
mut_pct_diffs = mut_cluster_mean.sort_index().values - mut_archetype_pct.sort_index().values[:,None]
mut_pct_diffs = pd.DataFrame(mut_pct_diffs, index=mut_cluster_mean.index, columns=mut_cluster_mean.columns)

#%%
# Plot the heatmap
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(wt_pct_diffs, cmap='viridis', aspect='auto')
axs[0].set_title('Wildtype')
axs[0].set_xticks(wt_pct_diffs.columns)
nyticks = len(wt_pct_diffs.index)
axs[0].set_yticks(range(nyticks), wt_cluster_mean.index.to_list())
axs[1].imshow(mut_pct_diffs, cmap='viridis', aspect='auto')
axs[1].set_title('Mutant')
axs[1].set_xticks(mut_pct_diffs.columns)
nyticks = len(mut_pct_diffs.index)
axs[1].set_yticks(range(nyticks), mut_pct_diffs.index.to_list())
# Add a label below the x axis ticks
axs[0].set_xlabel('Cluster', fontsize=18)
axs[1].set_xlabel('Cluster', fontsize=18)
# Add a label to the left of the y axis ticks
axs[0].set_ylabel('Archetype', fontsize=18)
# plt.colorbar()
plt.tight_layout()
# %%
# Find wildtype cells that are entirely within the polytope formed by wildtype archetypes
# This is a sanity check to make sure that the wildtype archetypes are well defined
wt_vertices = wt.uns['archetype_vertices'][:wt_n_archetypes-1]
archetype_dim = wt_vertices.shape[0]
mut_pca = mut.obsm['X_pca'][:, 0:wt_n_archetypes-1]
wt_pca = wt.obsm['X_pca'][:, 0:wt_n_archetypes-1]
in_out = []
for i in range(wt_pca.shape[0]):
    if i % 1000 == 0:
        print(f'Cell {i}')
    inside = polytope.is_inside(wt_vertices, wt_pca[i], delta=.1)
    in_out.append(inside)
in_out = np.array(in_out)
print('Number of WT cells inside wildtype archetypes:', np.sum(in_out))
# %%
# Scatter plot with points colored by whether they are inside or outside the wildtype archetypes
fig, ax = plt.subplots(1,1, figsize=(5,5))
pc1 = 0
pc2 = 1
ax.scatter(wt_pca[:,pc1][~in_out], wt_pca[:,pc2][~in_out], c='grey', s=.1, alpha=.8)
ax.scatter(wt_pca[:,pc1][in_out], wt_pca[:,pc2][in_out], c='magenta', s=3.9)
ax.scatter(wt_vertices[0,], wt_vertices[1,], c='blue', s=100)
ax.set_xlabel(f'PC{pc1}')
ax.set_ylabel(f'PC{pc2}')
# %%
# Repeat the analysis for the mutant cells in the wildtype archetype polytope
mut_in_out = []
for i in range(mut_pca.shape[0]):
    if i % 1000 == 0:
        print(f'Cell {i}')
    inside = polytope.is_inside(wt_vertices, mut_pca[i])
    mut_in_out.append(inside)
mut_in_out = np.array(mut_in_out)
print('Number of mutant cells inside wildtype archetypes:', np.sum(in_out))
# %%
# Scatter plot with points colored by whether they are inside or outside the wildtype archetypes
fig, ax = plt.subplots(1,1, figsize=(5,5))
pc1 = 0
pc2 = 1
ax.scatter(mut_pca[:,pc1][~mut_in_out], mut_pca[:,pc2][~mut_in_out], c='grey', s=.1, alpha=.8)
ax.scatter(mut_pca[:,pc1][mut_in_out], mut_pca[:,pc2][mut_in_out], c='magenta', s=3.9)
ax.scatter(wt_vertices[0,], wt_vertices[1,], c='blue', s=100)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
# %%
