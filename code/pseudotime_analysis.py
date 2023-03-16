#%%
import pandas as pd
#%%
data = pd.read_csv('../data/raw-counts-mes-wildtype.csv','r')

#%%
data_genes = pd.read_csv(foldername+'Bcell_markergenes.csv')
data_genes = data_genes.drop(['Unnamed: 0'], axis=1)#cell
true_label = data['time_hour']
data = data.drop(['cell', 'time_hour'], axis=1)
adata = sc.AnnData(data_genes)
adata.obsm['X_pca'] = data.values

# use UMAP or PHate to obtain embedding that is used for single-cell level visualization
embedding = umap.UMAP(random_state=42, n_neighbors=15, init='random').fit_transform(data.values[:, 0:5])
print('finished embedding')
# list marker genes or genes of interest if known in advance. otherwise marker_genes = []
marker_genes = ['Igll1', 'Myc', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4', 'Sp7']  # irf4 down-up
# call VIA. We identify an early (suitable) start cell root = [42]. Can also set an arbitrary value
via.via_wrapper(adata, true_label, embedding, knn=knn, ncomps=ncomps, jac_std_global=0.15, root=[42], dataset='',
            random_seed=1,v0_toobig=0.3, v1_toobig=0.1, marker_genes=marker_genes, piegraph_edgeweight_scalingfactor=1, piegraph_arrow_head_width=.1)
