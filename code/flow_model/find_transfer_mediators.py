#%%
import pickle
import sys
sys.path.append('..')
import scanpy as sc
import glob
from mediators import find_bridges
from collections import Counter

#%%
# Load the data
source_genotype = 'mutant'
target_genotype = 'wildtype'
# label = 'VIM_first_'
label = ''

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

node_to_idx = pickle.load(open(f'../../data/protein_id_to_idx.pickle', 'rb'))
idx_to_node = {v:k for k,v in node_to_idx.items()}
protein_id_name = pickle.load(open(f'../../data/protein_id_to_name.pickle', 'rb'))
protein_id_name = {k:'/'.join(v) for k,v in protein_id_name.items()}
all_genes = set(node_to_idx.keys())
protein_name_id = {v:k for k,v in protein_id_name.items() if k in all_genes}
knowledge_graph = pickle.load(open(f'../../data/filtered_graph.pickle', 'rb'))
tgt_graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
src_graph = pickle.load(open(f'{src_outdir}/optimal_{source_genotype}_graph.pickle', 'rb'))

#%%
best_gene_combinations = []
for file in glob.glob(f'{datadir}/top_{label}{transfer}_combination*.pickle'):
    combo = pickle.load(open(file, 'rb'))
    best_gene_combinations.append(combo)

# %%
sig_bridges = Counter()
for combo in best_gene_combinations:
    subgraph = src_graph.subgraph(combo)
    # Convert the subgraph to a dictionary of target gene: in edge genes
    subgraph_dict = {}
    for edge in subgraph.edges:
        source, target = edge
        if target not in subgraph_dict:
            subgraph_dict[target] = set()
        subgraph_dict[target].add(source)

    results  = find_bridges(subgraph_dict, 
                            knowledge_graph=knowledge_graph,
                            #all_shortest_paths=None,
                            verbose=False,
                            threshold=0.01)
    bridge_probs, bridged_interactions = results

    for bridge, p in bridge_probs.items():
        print(protein_id_name[bridge], p)
        sig_bridges[bridge] += 1
    print('-'*80)
        
# %%
for bridge, count in sig_bridges.most_common():
    print(f'{protein_id_name[bridge]:10s} {count:2d}/{len(best_gene_combinations)}')
# %%
