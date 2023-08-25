#%%
import pickle
import scanpy as sc
import numpy as np
from scipy.stats import hypergeom, ttest_ind
# %%
# tmstp = '20230608_093734'
tmstp = '20230607_165324'
genotype = 'wildtype'
mut = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')
cell_types = {c:i for i,c in enumerate(set(mut.obs['cell_type']))}
outdir = f'../../output/{tmstp}'
ko_dir = f'{outdir}/knockout_simulations'
pltdir = f'{outdir}/knockout_simulations/figures'
#%%
node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
all_genes = set(node_to_idx.keys())
# Convert from ids to gene names
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle','rb'))
protein_id_name = {id: '/'.join(name) for id, name in protein_id_name.items()}
name_protein_id = {name: id for id, name in protein_id_name.items()}

# %%
# Load the shortest paths
optimal_model_shortest_paths_graph = pickle.load(open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths_graph.pickle', 'rb'))
shortest_paths_to_target = pickle.load(open(f'../../output/{tmstp}/optimal_{genotype}_shortest_paths.pickle', 'rb'))
#%%
shortest_paths_from_source = {}
for target, paths in shortest_paths_to_target.items():
    for path in paths:
        source = path[0]
        if source not in shortest_paths_from_source:
            shortest_paths_from_source[source] = []
        shortest_paths_from_source[source].append(path)

#%%
all_shortest_paths = pickle.load(open(f'../../output/{tmstp}/all_shortest_paths.pickle', 'rb'))

all_shortest_paths_from_source = {}
for target, paths in all_shortest_paths.items():
    for path in paths:
        source = path[0]
        if source not in all_shortest_paths_from_source:
            all_shortest_paths_from_source[source] = []
        all_shortest_paths_from_source[source].append(path)


#%%
# Calculate the percentage of shortest paths that a mediator appears in for each knockout gene
def count_mediators(all_paths):
    mediators = {}
    target_counts = {}
    for source, paths in all_paths.items():
        for path in paths:
            if len(path) > 2:
                if source not in mediators:
                    mediators[source] = {}
                target = path[-1]
                if target not in mediators[source]:
                    mediators[source][target] = set()
                for mediator in path[1:-1]:
                    mediators[source][target].add(mediator)
        if source in mediators:
            target_counts[source] = len(mediators[source])

    mediator_counts = {}
    for source, target_mediators in mediators.items():
        mediator_counts[source] = {}
        for target, _mediators in target_mediators.items():
            for mediator in _mediators:
                if mediator not in mediator_counts[source]:
                    mediator_counts[source][mediator] = 0
                mediator_counts[source][mediator] += 1

    return mediator_counts, target_counts

#%%
# k - number of matches in chosen set, i.e. number of shortest paths that a mediator appears in
# M - total number of items, 
# n - number of matches in population
# N - size of set chosen at random
mediator_counts, target_counts = count_mediators(shortest_paths_from_source)
total_mediator_counts, _ = count_mediators(all_shortest_paths_from_source)
total_unique_paths = len(node_to_idx)**2
mediator_probs = {}
for source, mediators in mediator_counts.items():
    for mediator, count in mediators.items():
        # number of times we see this mediator in any shortest path from this source to the chosen targets
        k = mediator_counts[source][mediator]
        # Total possible number of shortest paths
        M = len(node_to_idx)
        # Number of times we see this mediator in any shortest path from the source to any target
        n = total_mediator_counts[source][mediator]
        # Number of targets of this source
        N = target_counts[source]
        p = hypergeom.sf(k, M, n, N)
        mediator_probs[(source, mediator)] = p
        print(protein_id_name[source], protein_id_name[mediator], p)
        print('k=',k)
        print('M=',M)
        print('n=',n)
        print('N=',N)
mediator_probs = {k: v for k, v in sorted(mediator_probs.items(), key=lambda item: item[1])}
#%%
significant_mediators = {}
mediated_sources = {}
num_significant = 0
for source_mediator, prob in mediator_probs.items():
    source, mediator = source_mediator
    # Divide by the number of tests for a Bonferroni correction
    if prob < 0.05/len(mediator_probs):
        if source not in significant_mediators:
            significant_mediators[source] = []
        if mediator not in mediated_sources:
            mediated_sources[mediator] = []
        significant_mediators[source].append(mediator)
        mediated_sources[mediator].append(source)
        num_significant += 1
        print(protein_id_name[source], protein_id_name[mediator], prob)
print(f'Number of significant mediators: {num_significant} out of {len(mediator_probs)}')
#%%
cell_type_ko_proportions = {}

for i,ko_gene in enumerate(all_genes):
    ko_gene_name = protein_id_name[ko_gene]
    # Load the knockout results
    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_proportions_mutant.pickle', 'rb') as f:
        ko_cell_type_proportions = pickle.load(f)
    perturb_cell_proportions, baseline_cell_proportions = ko_cell_type_proportions

    with open(f'{ko_dir}/{ko_gene_name}_knockout_cell_type_proportions_mutant.pickle', 'rb') as f:
        cell_type_ko_proportions[ko_gene] = {}
        for i,cell_type in enumerate(cell_types):
            cell_type_ko_proportions[ko_gene][cell_type] = perturb_cell_proportions[i]

# %%
# Compare the distribution of cell type changes in significant versus non-significant mediators
sig_mediator_cell_type_proportion_probabilities = {}
for cell_type in cell_types:
    for mediator, sources in mediated_sources.items():
        # Get the cell type proportions for the knockout genes that are 
        # associated with this mediator
        sig_proportions = np.zeros(len(sources))
        for i,source in enumerate(sources):
            sig_proportions[i] = cell_type_ko_proportions[source][cell_type]
        # Get the cell type proportions for the knockout genes that are NOT mediated 
        # by this mediator
        non_sig_proportions = np.zeros(len(all_genes) - len(sources))
        i=0
        for source in all_genes:
            if source not in sources:
                non_sig_proportions[i] = cell_type_ko_proportions[source][cell_type]
                i += 1
        # Check that we have the right number of non-significant genes
        if i!=len(non_sig_proportions):
            raise Exception('i!=len(non_sig_proportions)')
        
        # Ensure that we have enough data points to calculate a p-value
        if len(sig_proportions) <= 1 or len(non_sig_proportions) <= 1:
            continue
        # Calculate a p-value for the difference in means
        result = ttest_ind(sig_proportions, non_sig_proportions)
        p = result.pvalue

        sig_mediator_cell_type_proportion_probabilities[(mediator, cell_type)] = (
              np.mean(sig_proportions),
              np.mean(non_sig_proportions),
              p)
        
# %%
# Sort by the difference in means
sorted_sig_mediator_cell_type_proportion_probabilities = \
    {k: v for k, v in sorted(sig_mediator_cell_type_proportion_probabilities.items(), key=lambda item: abs(item[1][0] - item[1][1]), reverse=True)}
for (mediator, cell_type), (cell_type_proportions_sig, cell_type_proportions_nonsig, p) in sorted_sig_mediator_cell_type_proportion_probabilities.items():
    # Get the list of genes that are significantly mediated by this mediator
    mediated_genes = mediated_sources[mediator]
    if p < 0.05:
        print(f'{protein_id_name[mediator]:10s} {cell_type:5s} '
                f'{np.mean(cell_type_proportions_sig): .3f} '
                f'{np.mean(cell_type_proportions_nonsig): .3f} '
                f'{p:.3f} ', 
                f'{",".join([protein_id_name[g] for g in mediated_genes])}')


# %%
