#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt

#%%
# Load the list of shortest paths
genotype = 'mutant'
outdir = f'../../output/20230608_093734'
shortest_paths = pickle.load(open(f'{outdir}/shortest_paths_{genotype}.pickle', 'rb'))

# %%
path_lens = []
for target in shortest_paths:
    for path in shortest_paths[target]:
        path_lens.append(len(path))
# %%
plt.hist(path_lens, bins=range(1, max(path_lens)+1))
# %%
print(f'Average path length: {np.mean(path_lens):.2f}')
print(f'Median path length: {np.median(path_lens):.2f}')
print(f'Max path length: {max(path_lens)}')

