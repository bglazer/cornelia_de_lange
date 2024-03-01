#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import glob
import networkx as nx
import matplotlib

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

    tgt_graph = pickle.load(open(f'{tgt_outdir}/optimal_{target_genotype}_graph.pickle', 'rb'))
    src_graph = pickle.load(open(f'{src_outdir}/optimal_{source_genotype}_graph.pickle', 'rb'))

    # Get the subgraph of the two graphs that contains the combination genes
    print(f'Experiment: {experiment}')
    centrality_diff = {}
    for combo in best_gene_combinations[experiment]:
        tgt_subgraph = tgt_graph.subgraph(combo)
        src_subgraph = src_graph.subgraph(combo)
        # Compare the centrality of the nodes in the two subgraphs
        tgt_centrality = nx.betweenness_centrality(tgt_graph)
        src_centrality = nx.betweenness_centrality(src_graph)
        for node in combo:
            if node not in tgt_centrality:
                tgt_centrality[node] = 0
            if node not in src_centrality:
                src_centrality[node] = 0
            if node not in centrality_diff:
                centrality_diff[node] = []
            centrality_diff[node].append(tgt_centrality[node] - src_centrality[node])
    # Sort the nodes by the difference in centrality
    # sorted_nodes = sorted(centrality_diff.items(), key=lambda x: np.mean(x[1]), reverse=True)
    # for gene, diff in sorted_nodes:
    #     print(f'{protein_id_name[gene]}: {np.mean(diff)}')
    # print('-------------------')

    # Get the significant genes
    significant_genes = pickle.load(open(f'{datadir}/{label}{transfer}_significant_genes.pickle', 'rb'))
    tgt_subgraph = tgt_graph.subgraph(significant_genes)
    src_subgraph = src_graph.subgraph(significant_genes)
    subgraph = nx.compose(tgt_subgraph, src_subgraph)
    layout = nx.kamada_kawai_layout(subgraph)

    # Get the edges in the tgt subgraph that are not in the src subgraph
    src_edges = set(src_subgraph.edges)
    tgt_edges = set(tgt_subgraph.edges)
    src_only_edges = tgt_edges - src_edges
    tgt_only_edges = src_edges - tgt_edges
    common_edges = src_edges & tgt_edges
    # Remove self loops
    self_loops = set([(node, node) for node in subgraph.nodes])
    common_edges -= self_loops
    src_only_graph = nx.DiGraph()
    src_only_graph.add_edges_from(src_only_edges)
    src_only_graph.add_nodes_from(subgraph.nodes)
    tgt_only_graph = nx.DiGraph()
    tgt_only_graph.add_edges_from(tgt_only_edges)
    tgt_only_graph.add_nodes_from(subgraph.nodes)
    common_graph = nx.DiGraph()
    common_graph.add_edges_from(common_edges)
    common_graph.add_nodes_from(subgraph.nodes)
    
    # Color nodes by their connectivity
    colormap = plt.get_cmap('coolwarm')
    centrality_diff = np.array([np.mean(centrality_diff[node]) for node in subgraph.nodes])
    # Scale the centrality difference to the range [0, 1]
    centrality_scaled = (centrality_diff - centrality_diff.min()) / (centrality_diff.max() - centrality_diff.min())
    node_colors = [colormap(centrality_scaled[i]) for i in range(len(subgraph.nodes))]
    # Increase the dpi of the figure so the text is not blurry
    fig = plt.figure(dpi=300)
    nx.draw_networkx_nodes(
            subgraph, pos=layout,
            # Change the node border color to blue
            node_color=node_colors,
            # alpha=node_scale+0.5,
            # with_labels=False,
            )
    nx.draw_networkx_edges(
            common_graph, pos=layout,
            edge_color='#d3d3d339',
            width=1,
            )
    nx.draw_networkx_edges(
            src_only_graph, pos=layout,
            edge_color='#1f77b439',
            width=1,
            )
    nx.draw_networkx_edges(
            tgt_only_graph, pos=layout,
            edge_color='#ff7f0e39',
            width=1,
            )

    plt_height = plt.ylim()[1] - plt.ylim()[0]
    # Get figure size in inches and dpi
    fig_height_in = fig.get_figheight()
    dpi = fig.get_dpi()
    # Convert node size from points**2 to the fig data scale
    node_size_in = .5
    # Convert node size from fig data scale to axes data scale
    node_size_ax = node_size_in*2 * plt_height / fig_height_in

    for i in range(len(subgraph.nodes)):
        gene = list(subgraph.nodes)[i]
        plt.text(layout[gene][0], layout[gene][1], protein_id_name[gene], 
                 horizontalalignment='center', verticalalignment='top',
                 fontsize=8, color='black',
                 bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.1'))
    # plt.title(title)
    centrality_max = max(abs(centrality_diff.min()), abs(centrality_diff.max()))
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-centrality_max, vmax=centrality_max),
                                            cmap=colormap),
                                            shrink=.5, label='Centrality Difference');
    plt.title(f'{source_genotype.capitalize()} to {target_genotype.capitalize()} {"" if label == "" else "(VIM first)"}')
    # Add a legend annotating the edge colors
    labels = ['Common edges', 
                f'{source_genotype.capitalize()} only edges', 
                f'{target_genotype.capitalize()} only edges']
    artists = [plt.Line2D([0], [0], color='#d3d3d339', lw=2),
                plt.Line2D([0], [0], color='#1f77b439', lw=2),
                plt.Line2D([0], [0], color='#ff7f0e39', lw=2)]
    plt.legend(artists, labels, bbox_to_anchor=(1.05, 1.05), loc='upper left',
               fontsize='small', title='Edge type')
    plt.show()
    # break

# %%
