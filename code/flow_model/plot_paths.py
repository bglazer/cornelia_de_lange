#%%
import matplotlib.pyplot as plt
import pickle
import numpy as np
from itertools import combinations, combinations_with_replacement
from mip import Model, MINIMIZE, BINARY, xsum

#%%
wt_tmstp = '20230607_165324'  
mut_tmstp = '20230608_093734'
outdir = f'../../output/'
#%%
wt_paths = pickle.load(open(f'{outdir}/{wt_tmstp}/shortest_paths_wildtype.pickle', 'rb'))
# %%
protein_id_name = pickle.load(open('../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {pid: '/'.join(names) for pid, names in protein_id_name.items()}
protein_name_id = {protein_id_name[pid]: pid for pid in wt_paths.keys()}
#%%
pid = protein_name_id['NANOG']
paths = wt_paths[pid]
# %%


#%%
optimized_placement = optimize_placement(levels, max_count=None, verbose=True)
#%%
plot_paths(optimized_placement, paths, center=True)
# %%
