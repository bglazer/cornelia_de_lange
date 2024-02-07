#%%
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import numpy as np
from util import distinct_colors
import scanpy as sc
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import ttest_ind
#%%
wt_tmstp = '20230607_165324'
mut_tmstp = '20230608_093734'
wt_data = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mut_data = sc.read_h5ad(f'../../data/mutant_net.h5ad')
wt_X = wt_data.X.toarray()
mut_X = mut_data.X.toarray()
combo_X = np.concatenate([wt_X, mut_X], axis=0)
# %%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}# %%

# %%
# %%
pca = PCA()
# Set the PC mean and components
pca.mean_ = wt_data.uns['pca_mean']
pca.components_ = wt_data.uns['PCs']

wt_X_pca = pca.transform(wt_X)
mut_X_pca = pca.transform(mut_X)
#%%

def expression_grid(gene_id, nbins=50, p_thresh=0.05):
    gene_name = protein_id_name[gene_id]
    wt_gene_X = wt_X[:,node_to_idx[gene_id]]
    mut_gene_X = mut_X[:,node_to_idx[gene_id]]

    # Make a grid in PCA space
    import numpy as np
    X_pca = np.concatenate([wt_X_pca, mut_X_pca], axis=0)
    buf = (X_pca.max() - X_pca.min())/nbins
    x = np.linspace(X_pca[:,0].min()-buf, X_pca[:,0].max()+buf, nbins)
    y = np.linspace(X_pca[:,1].min()-buf, X_pca[:,1].max()+buf, nbins)

    # Assign cells in PCA space to a grid
    w = x[1]-x[0]
    h = y[1]-y[0]
    wt_xgrid_idx = np.where((wt_X_pca[:,0,None] > x) & (wt_X_pca[:,0,None] < (x+w)))[1]
    wt_ygrid_idx = np.where((wt_X_pca[:,1,None] > y) & (wt_X_pca[:,1,None] < (y+h)))[1]
    mut_xgrid_idx = np.where((mut_X_pca[:,0,None] > x) & (mut_X_pca[:,0,None] < (x+w)))[1]
    mut_ygrid_idx = np.where((mut_X_pca[:,1,None] > y) & (mut_X_pca[:,1,None] < (y+h)))[1]
    # Compute the WT and MUT expression in each grid
    wt_gene_grid = np.zeros((nbins,nbins))
    wt_grid_count = np.zeros((nbins,nbins))
    mut_gene_grid = np.zeros((nbins,nbins))
    mut_grid_count = np.zeros((nbins,nbins))
    for i in range(nbins):
        for j in range(nbins):
            wt_grid_count[i,j] = np.sum((wt_xgrid_idx==i) & (wt_ygrid_idx==j))
            mut_grid_count[i,j] = np.sum((mut_xgrid_idx==i) & (mut_ygrid_idx==j))
    for i in range(nbins):
        for j in range(nbins):
            if wt_grid_count[i,j] > 0:
                wt_gene_grid[i,j] = np.mean(wt_gene_X[(wt_xgrid_idx==i) & (wt_ygrid_idx==j)])
            if mut_grid_count[i,j] > 0:
                mut_gene_grid[i,j] = np.mean(mut_gene_X[(mut_xgrid_idx==i) & (mut_ygrid_idx==j)])
    # Compute the statistical significance of the difference in means in each grid, using a t-test
    from scipy.stats import ttest_ind
    pvals = np.ones((nbins,nbins))
    for i in range(nbins):
        for j in range(nbins):
            if wt_grid_count[i,j] > 0 and mut_grid_count[i,j] > 0:
                _, pvals[i,j] = ttest_ind(wt_gene_X[(wt_xgrid_idx==i) & (wt_ygrid_idx==j)], 
                                            mut_gene_X[(mut_xgrid_idx==i) & (mut_ygrid_idx==j)])
    sig_grids = np.where(pvals < p_thresh)
    # Get the cell types of the cells in each grid
    wt_cell_types = np.array(wt_data.obs['cell_type'])
    mut_cell_types = np.array(mut_data.obs['cell_type'])
    combo_cell_types = np.concatenate([wt_cell_types, mut_cell_types], axis=0)
    cell_type_to_idx = {cell_type: i for i,cell_type in enumerate(np.unique(combo_cell_types))}
    cell_type_to_idx = {cell_type: i for cell_type,i in sorted(cell_type_to_idx.items())}

    sig_cell_types = []
    for i in range(nbins):
        for j in range(nbins):
            if pvals[i,j] < p_thresh:
                sig_cell_types.extend(wt_cell_types[(wt_xgrid_idx==i) & (wt_ygrid_idx==j)])
                sig_cell_types.extend(mut_cell_types[(mut_xgrid_idx==i) & (mut_ygrid_idx==j)])
    sig_cell_type_counts = Counter(sig_cell_types)
    # Compute the probability of getting the observed number of significant cells of each type
    n_repeats = 1000
    rand_cell_type_counts = {cell_type: [] for cell_type in cell_type_to_idx.keys()}
    nonzero_grids = np.where((wt_grid_count > 0) | (mut_grid_count > 0))
    for i in range(n_repeats):
        # Select a random subset of grids from the WT and MUT 
        # grid of size equal to the number of significant grids
        rand_grid_idxs = np.random.choice(range(len(nonzero_grids[0])), size=len(sig_grids[0]), replace=False)
        rand_grids = (nonzero_grids[0][rand_grid_idxs], nonzero_grids[1][rand_grid_idxs])
        rand_cell_types = []
        for i,j in zip(*rand_grids):
            rand_cell_types.extend(wt_cell_types[(wt_xgrid_idx==i) & (wt_ygrid_idx==j)])
            rand_cell_types.extend(mut_cell_types[(mut_xgrid_idx==i) & (mut_ygrid_idx==j)])
        # Compute the number of random cells of each type
        cell_type_counts = Counter(rand_cell_types)
        for cell_type in rand_cell_type_counts:
            rand_cell_type_counts[cell_type].append(cell_type_counts[cell_type])

    # Compute the probability of getting the observed number of significant cells of each type
    pvals_cell_type = {}
    for cell_type in cell_type_to_idx.keys():
        pvals_cell_type[cell_type] = np.sum(np.array(rand_cell_type_counts[cell_type]) >= sig_cell_type_counts[cell_type])/n_repeats
        p = pvals_cell_type[cell_type]
        print(f'{cell_type:6s}: {p:.4f} {"*" if p < 0.05 else ""}')
    # return

    fig, axs = plt.subplots(5,2, figsize=(15,20))

    count_max = np.max([wt_grid_count, mut_grid_count])
    expr_max = np.max([wt_gene_grid, mut_gene_grid])
    
    r00 = axs[0,0].scatter(wt_X_pca[:,0], wt_X_pca[:,1], s=1, alpha=0.6, 
                            c=wt_gene_X, vmin=0, vmax=expr_max)
    r01 = axs[0,1].scatter(mut_X_pca[:,0], mut_X_pca[:,1], s=1, alpha=0.6, 
                            c=mut_gene_X, vmin=0, vmax=expr_max)
    axs[0,0].set_title(f'WT {gene_name} expression')
    axs[0,1].set_title(f'Mutant {gene_name} expression')
    
    r10 = axs[1,0].imshow((wt_grid_count[:,::-1].T), 
                    cmap='Oranges',
                    interpolation='none', aspect='auto', 
                    vmin=0, vmax=count_max)
    r11 = axs[1,1].imshow((mut_grid_count[:,::-1].T), 
                    cmap='Oranges',
                    interpolation='none', aspect='auto',
                    vmin=0, vmax=count_max)
    axs[1,0].set_title('WT grid cell count')
    axs[1,1].set_title('Mutant grid cell count')

    r20 = axs[2,0].imshow(wt_gene_grid[:,::-1].T, cmap='viridis', 
                            interpolation='none', aspect='auto',
                            vmin=0, vmax=expr_max)
    r21 = axs[2,1].imshow(mut_gene_grid[:,::-1].T, cmap='viridis', 
                            interpolation='none', aspect='auto',
                            vmin=0, vmax=expr_max)
    axs[2,0].set_title(f'WT {gene_name} expression')
    axs[2,1].set_title(f'Mutant {gene_name} expression')
    # Use a diverging colormap to show the difference, centered at zero
    cmap = plt.get_cmap('RdBu_r')

    diff = (wt_gene_grid-mut_gene_grid)
    # if the grid count is zero, make the difference zero
    diff[wt_grid_count<=3] = 0
    diff[mut_grid_count<=3] = 0
    max_diff = np.max(np.abs(diff))

    r30 = axs[3,0].imshow(diff[:,::-1].T,
                            cmap=cmap,
                            interpolation='none', aspect='auto',
                            vmin=-max_diff, vmax=max_diff)
                            
    axs[3,0].set_title(f'{gene_name} WT - Mutant difference')
    # Plot the dominant cell type for each grid
    wt_cell_type_grid = np.zeros((nbins,nbins), dtype=int)
    mut_cell_type_grid = np.zeros((nbins,nbins), dtype=int)
    for i in range(nbins):
        for j in range(nbins):
            if wt_grid_count[i,j] > 0:
                wt_grid_types = wt_cell_types[(wt_xgrid_idx==i) & (wt_ygrid_idx==j)]
                wt_grid_type_idx_counts = np.zeros(len(cell_type_to_idx))
                for cell_type in wt_grid_types:
                    type_idx = cell_type_to_idx[cell_type]
                    wt_grid_type_idx_counts[type_idx] += 1
                wt_grid_idx_max = np.argmax(wt_grid_type_idx_counts)
                wt_cell_type_grid[i,j] = wt_grid_idx_max
            if mut_grid_count[i,j] > 0:
                mut_grid_types = mut_cell_types[(mut_xgrid_idx==i) & (mut_ygrid_idx==j)]
                mut_grid_type_idx_counts = np.zeros(len(cell_type_to_idx))
                for cell_type in mut_grid_types:
                    type_idx = cell_type_to_idx[cell_type]
                    mut_grid_type_idx_counts[type_idx] += 1
                mut_grid_idx_max = np.argmax(mut_grid_type_idx_counts)
                mut_cell_type_grid[i,j] = mut_grid_idx_max

    cmap = distinct_colors(len(cell_type_to_idx))

    wt_cell_type_colors = np.zeros((nbins,nbins,4), dtype=int)
    mut_cell_type_colors = np.zeros((nbins,nbins,4), dtype=int)
    for i in range(nbins):
        for j in range(nbins):
            if wt_grid_count[i,j] > 0:
                hex_color = cmap[wt_cell_type_grid[i,j]]
                rgba_color = tuple(int(hex_color[i:i+2], 16) for i in (1,3,5)) + (255,)
                wt_cell_type_colors[i,j] = rgba_color
            else:
                wt_cell_type_colors[i,j] = (0,0,0,0)

            if mut_grid_count[i,j] > 0:
                hex_color = cmap[mut_cell_type_grid[i,j]]
                rgba_color = tuple(int(hex_color[i:i+2], 16) for i in (1,3,5)) + (255,)
                mut_cell_type_colors[i,j] = rgba_color
            else:
                mut_cell_type_colors[i,j] = (0,0,0,0)

    axs[4,0].imshow(wt_cell_type_colors[:,::-1].transpose(1,0,2),
                    interpolation='none', aspect='auto') 
    axs[4,1].imshow(mut_cell_type_colors[:,::-1].transpose(1,0,2),
                    interpolation='none', aspect='auto') 
    # Make axs[4,0] subplot the same size as axs[3,0]   
    
    axs[4,0].legend([plt.Rectangle((0,0),1,1,fc=cmap[i]) for c,i in cell_type_to_idx.items()], 
                    cell_type_to_idx.keys(), 
                    loc='center left', 
                    bbox_to_anchor=(1,0.5))
    axs[4,1].legend([plt.Rectangle((0,0),1,1,fc=cmap[i]) for c,i in cell_type_to_idx.items()], 
                    cell_type_to_idx.keys(), 
                    loc='center left', 
                    bbox_to_anchor=(1,0.5))
    axs[4,0].set_title('WT grid cell most common cell type')
    axs[4,1].set_title('Mutant grid cell most common cell type')

    # Annotate the axs[3,0] grid with p-values less than the threshold
    for i in range(nbins):
        for j in range(nbins):
            if pvals[:,::-1].T[i,j] < p_thresh:
                axs[1,0].text(j,i, 'x', fontsize=10, color='black', ha='center', va='center', weight='bold')
                axs[1,1].text(j,i, 'x', fontsize=10, color='black', ha='center', va='center', weight='bold')
                axs[2,0].text(j,i, 'x', fontsize=10, color='orange', ha='center', va='center', weight='bold')
                axs[2,1].text(j,i, 'x', fontsize=10, color='orange', ha='center', va='center', weight='bold')
                axs[3,0].text(j,i, 'x', fontsize=10, ha='center', va='center', weight='bold')
                axs[4,0].text(j,i, 'x', fontsize=10, ha='center', va='center', c='white', weight='bold')
                axs[4,1].text(j,i, 'x', fontsize=10, ha='center', va='center', c='white', weight='bold')

    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    # Add a colorbar to each axis
    plt.colorbar(r00, ax=axs[0,0])
    plt.colorbar(r01, ax=axs[0,1])
    plt.colorbar(r10, ax=axs[1,0])
    plt.colorbar(r11, ax=axs[1,1])
    plt.colorbar(r20, ax=axs[2,0])
    plt.colorbar(r21, ax=axs[2,1])
    plt.colorbar(r30, ax=axs[3,0])
    # plt.colorbar(r31, ax=axs[3,1])
    fig.tight_layout()

def overall_expression(gene_name):
    # Make a violin plot of overall expression of the gene
    gene_id = protein_name_id[gene_name]
    wt_gene_X = wt_X[:,node_to_idx[gene_id]]
    mut_gene_X = mut_X[:,node_to_idx[gene_id]]
    fig, ax = plt.subplots()
    ax.violinplot([wt_gene_X, mut_gene_X])
    ax.set_xticks([1,2])
    ax.set_xticklabels(['WT', 'Mutant'])
    ax.set_ylabel(f'{gene_name} expression')
    _, p = ttest_ind(wt_gene_X, mut_gene_X)
    ax.set_title(f'{gene_name} p={p:.2}')
    # Add a horizontal line segment at the mean expression of each group
    ax.plot([1-0.1, 1+0.1], [np.mean(wt_gene_X), np.mean(wt_gene_X)], c='black', alpha=0.5)
    ax.plot([2-0.1, 2+0.1], [np.mean(mut_gene_X), np.mean(mut_gene_X)], c='black', alpha=0.5)
    fig.tight_layout()

def cell_type_expression(gene_name):
    # Compare gene expression in each cell type
    gene_id = protein_name_id[gene_name]
    # Get a set of cell types
    wt_cell_types = np.array(wt_data.obs['cell_type'])
    mut_cell_types = np.array(mut_data.obs['cell_type'])
    wt_cell_type_set = set(wt_data.obs['cell_type'])
    mut_cell_type_set = set(mut_data.obs['cell_type'])
    cell_types = wt_cell_type_set.union(mut_cell_type_set)
    cell_types = sorted(list(cell_types))
    # Get the expression of the gene in each cell type
    wt_cell_type_expression = {cell_type: [] for cell_type in cell_types}
    mut_cell_type_expression = {cell_type: [] for cell_type in cell_types}
    for cell_type in cell_types:
        wt_cell_type_expression[cell_type] = wt_X[wt_cell_types==cell_type,node_to_idx[gene_id]]
        mut_cell_type_expression[cell_type] = mut_X[mut_cell_types==cell_type,node_to_idx[gene_id]]
    # Make a violin plot of the expression in each cell type
    fig, axs = plt.subplots(1,1, figsize=(15,5))
    # Put the WT and MUT expression side by side
    w = 1.5
    wt_x = np.arange(len(cell_types))*w
    mut_x = wt_x + w/2
    wt_violin = axs.violinplot([wt_cell_type_expression[cell_type] for cell_type in cell_types], positions=wt_x)
    mut_violin = axs.violinplot([mut_cell_type_expression[cell_type] for cell_type in cell_types], positions=mut_x)
    axs.set_xticks(wt_x+w/4)
    # Add vertical lines between each cell type
    for x in wt_x[1:]:
        axs.axvline(x-w/4, color='black', alpha=0.5)
    # Add horizontal line segments at the mean expression of each cell type
    for x,cell_type in zip(wt_x, cell_types):
        axs.plot([x-w/8, x+w/8], [np.mean(wt_cell_type_expression[cell_type]), np.mean(wt_cell_type_expression[cell_type])], c='black', alpha=0.5)
        axs.plot([x-w/8+w/2, x+w/8+w/2], [np.mean(mut_cell_type_expression[cell_type]), np.mean(mut_cell_type_expression[cell_type])], c='black', alpha=0.5)
    # Compute a p-value for the difference in expression in each cell type
    pvals = []
    for cell_type in cell_types:
        _, p = ttest_ind(wt_cell_type_expression[cell_type], mut_cell_type_expression[cell_type])
        pvals.append(p)
    # Add a star to the cell types with a p-value less than 0.05
    axs.set_xticklabels([f'{cell_types[i]}{"*" if pvals[i] <.05 else ""}\np={pvals[i]:.1}' for i in range(len(cell_types))])
    axs.set_ylabel(f'{gene_name} expression')
    axs.set_title(f'{gene_name} expression in each cell type')
    # Add a legend
    axs.legend([wt_violin['bodies'][0], mut_violin['bodies'][0]], ['WT', 'NIPBL +/-'])
    fig.tight_layout()


# %%
nbins = 30
p_thresh = 0.05
#%%
gene_id = protein_name_id['POU5F1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
#%%
overall_expression(gene_name='POU5F1')
#%%
cell_type_expression(gene_name='POU5F1')

#%%
gene_id = protein_name_id['VIM']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='VIM')
cell_type_expression(gene_name='VIM')

#%%
gene_id = protein_name_id['SNAI1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='SNAI1')
cell_type_expression(gene_name='SNAI1')
#%%
gene_id = protein_name_id['MESP1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='MESP1')
cell_type_expression(gene_name='MESP1')

# %%
gene_id = protein_name_id['ID3']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='ID3')
cell_type_expression(gene_name='ID3')
# %%
gene_id = protein_name_id['ID2']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='ID2')
cell_type_expression(gene_name='ID2')
#%%
gene_id = protein_name_id['FOXF1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='FOXF1')
cell_type_expression(gene_name='FOXF1')

#%%
gene_id = protein_name_id['MSX2']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='MSX2')
cell_type_expression(gene_name='MSX2')

#%%
gene_id = protein_name_id['CDX1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='CDX1')
cell_type_expression(gene_name='CDX1')

#%%
gene_id = protein_name_id['CDX2']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='CDX2')
cell_type_expression(gene_name='CDX2')

#%%
gene_id = protein_name_id['HOXB2']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
# %%
gene_id = protein_name_id['HOXB1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
# %%
overall_expression(gene_name='HOXB1')
cell_type_expression(gene_name='HOXB1')
# %%
gene_id = protein_name_id['NANOG']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='NANOG')
cell_type_expression(gene_name='NANOG')
# %%
gene_id = protein_name_id['CITED2']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='CITED2')
cell_type_expression(gene_name='CITED2')

# %%
gene_id = protein_name_id['TBX1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='TBX1')
cell_type_expression(gene_name='TBX1')
# %%
# %%
gene_id = protein_name_id['DKK1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='DKK1')
cell_type_expression(gene_name='DKK1')
# %%
gene_id = protein_name_id['MMP9']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='MMP9')
cell_type_expression(gene_name='MMP9')

# %%
gene_id = protein_name_id['PBX1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='PBX1')
cell_type_expression(gene_name='PBX1')

# %%
gene_id = protein_name_id['SLC9A3R1']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='SLC9A3R1')
cell_type_expression(gene_name='SLC9A3R1')
# %%
gene_id = protein_name_id['TBX6']
expression_grid(gene_id=gene_id, nbins=nbins, p_thresh=p_thresh)
overall_expression(gene_name='TBX6')
cell_type_expression(gene_name='TBX6')
# %%
