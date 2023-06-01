#%%
import pickle
import scanpy as sc
from collections import Counter
#%%
wt_tmstp = '20230526_105500'  
mut_tmstp = '20230530_093114'
outdir = f'../../output/'
#%%
adata = sc.read_h5ad(f'../../data/wildtype_net.h5ad')
mutant_inputs = pickle.load(open(f'{outdir}/{mut_tmstp}/input_selection_pvalues_mutant.pickle','rb'))
wildtype_inputs = pickle.load(open(f'{outdir}/{wt_tmstp}/input_selection_pvalues_wildtype.pickle','rb'))
#%%
wt_only = dict()
mut_only = dict()
wt_counts = Counter()
mut_counts = Counter()
for target_idx in wildtype_inputs:
    wt_selected = set([idx for idx, pval in wildtype_inputs[target_idx].items() if pval < .01])
    mut_selected = set([idx for idx, pval in mutant_inputs[target_idx].items() if pval < .01])
    in_wt_not_in_mut = wt_selected - mut_selected
    in_mut_not_in_wt = mut_selected - wt_selected
    
    for src_idx in in_wt_not_in_mut:
        if src_idx not in wt_only:
            wt_only[src_idx] = []
        wt_only[src_idx].append(target_idx)
    for src_idx in in_mut_not_in_wt:
        if src_idx not in mut_only:
            mut_only[src_idx] = []
        mut_only[src_idx].append(target_idx)

    wt_counts.update(wt_selected)
    mut_counts.update(mut_selected)

    if len(in_wt_not_in_mut) > 0 or len(in_mut_not_in_wt) > 0:
        print(f'For target {adata.var_names[target_idx]}:')
        if len(in_wt_not_in_mut) > 0:
            print('In wildtype but not in mutant:')
            for idx in in_wt_not_in_mut:
                print(f'{adata.var_names[idx]}')
        if len(in_mut_not_in_wt) > 0:    
            print('In mutant but not in wildtype:')
            for idx in in_mut_not_in_wt:
                print(f'{adata.var_names[idx]}')
        print('-'*40)
              

# %%
print('Genes that are most frequently in wildtype but not mutant')
for idx,targets in sorted(wt_only.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f'{adata.var_names[idx]:8s}: {len(targets):3d} {wt_counts[idx]}')

# # %%
print('Genes that are most frequently in mutant but not wildtype')
for idx,targets in sorted(mut_only.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f'{adata.var_names[idx]:8s}: {len(targets):3d} {mut_counts[idx]}')
# %%
# How does MESP1 change its targets from WT to mutant?
gene = 'MESP1'
gene_idx = adata.var_names.get_loc(gene)

for src_idx in wt_only:
    print(f'Genes targeted by {adata.var_names[src_idx]} in wildtype but not mutant')
    for target_idx in wt_only[src_idx]:
        print(f'{adata.var_names[target_idx]}')
#%%
for src_idx in mut_only:
    print(f'Genes targeted by {adata.var_names[src_idx]} in mutant but not wildtype')
    for target_idx in mut_only[src_idx]:
        print(f'{adata.var_names[target_idx]}')
# print('Genes targeted by gene in both wildtype and mutant')
# for idx in common_targets:
#     print(f'{adata.var_names[idx]}')
# %%
