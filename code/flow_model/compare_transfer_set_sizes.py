#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import glob
import networkx as nx

#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}

#%%
experiments = [
    ('mutant', 'wildtype', ''),
    ('wildtype', 'mutant', ''),
    ('wildtype', 'mutant', 'VIM_first_')
]
best_gene_combinations = {}
significant_genes = {}
p_components = {}

for experiment in experiments:
    source_genotype, target_genotype, label = experiment
    src_tmstp = '20230607_165324' if source_genotype == 'wildtype' else '20230608_093734'
    tgt_tmstp = '20230607_165324' if target_genotype == 'wildtype' else '20230608_093734'
    tgt_data = sc.read_h5ad(f'../../data/{target_genotype}_net.h5ad')
    src_data = sc.read_h5ad(f'../../data/{source_genotype}_net.h5ad')
    tgt_outdir = f'../../output/{tgt_tmstp}'
    src_outdir = f'../../output/{src_tmstp}'
    transfer = f'{source_genotype}_to_{target_genotype}'
    transfer_dir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations'
    pltdir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/figures'
    datadir = f'{tgt_outdir}/{transfer}_combination_transfer_simulations/data'
    
    best_gene_combinations[experiment] = []
    for file in glob.glob(f'{datadir}/top_{label}{transfer}_combination*.pickle'):
        combo = pickle.load(open(file, 'rb'))
        best_gene_combinations[experiment].append(combo)

    significant_genes[experiment] = pickle.load(open(f'{datadir}/{label}{transfer}_significant_genes.pickle', 'rb'))
    with open(f'{datadir}/{label}{transfer}_p_components.pickle', 'rb') as f:
        p_components[experiment] = pickle.load(f)

# %%
# TODO compare number of genes in combination to the size of the largest connected component
# formed by the genes in the combination
# Get the tab20 color map
cmap = plt.get_cmap('tab20')

largest_value = 0
for i, experiment in enumerate(experiments):
    source_genotype, target_genotype, label = experiment
    experiment_name = f'{source_genotype} to {target_genotype} {"" if label == "" else "(VIM first)"}'
    # Make a dot plot of the number of genes in the best combinations
    best_combinations = best_gene_combinations[experiment]
    len_combos = []
    len_ccs = []
    ps = []
    for len_combo, len_cc, p in p_components[experiment]:
        len_combos.append(len_combo)
        len_ccs.append(len_cc)
        ps.append(p)
    largest_value = max(largest_value, max(len_combos))
    
    x = np.linspace(i-1/4, i+1/4, len(len_combos))
    
    plt.scatter(x=x,
                y=len_combos, label=experiment_name,
                s=20, alpha=1, color=cmap(i*2), 
                marker='^')
    plt.scatter(x=x,
                y=len_ccs, label=experiment_name,
                s=20, alpha=1, color=cmap(i*2+1), 
                # Triangle marker
                marker='v')
    # Add an annotation if the p-value is significant
    for j, p in enumerate(ps):
        if p < 0.05:
            plt.text(x[j], len_combos[j]*1.05, '*', ha='center', va='bottom', fontsize=8)
    # Draw a line between the two points
    for j in range(len(len_combos)):
        plt.plot([x[j], x[j]], [len_combos[j], len_ccs[j]], 'k-', lw=0.5)

# plt.legend()
plt.xticks(range(len(experiments)), 
           [f'{e[0]} to {e[1]} {"" if e[2] == "" else "(VIM first)"}' 
            for e in experiments], 
            rotation=0)
plt.ylabel('Number of genes')
# Make a legend for the two marker types
plt.legend(['Transfer combination size', 'Largest connected component size'])
plt.ylim(0, largest_value*1.25)
#%%
# Make a Venn diagram with the significant genes in the best combinations
from matplotlib_venn_wordcloud import venn3_wordcloud
significant_genes_sets = []
experiment_labels = [f'{e[0]} to {e[1]} {"" if e[2] == "" else "(VIM first)"}' for e in experiments]

for experiment in experiments:
    gene_name_set = set()
    for gene in significant_genes[experiment]:
        gene_name_set.add(protein_id_name[gene])
    significant_genes_sets.append(gene_name_set)
venn3_wordcloud(significant_genes_sets,
                # set_labels = experiment_labels,
                set_colors = ('#FF5733', '#33FF57', '#5733FF'),
)

            
# %%
