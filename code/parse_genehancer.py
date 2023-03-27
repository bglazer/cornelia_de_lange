# %%
import pickle
graph = pickle.load(open('../data/filtered_graph.pickle','rb'))
protein_name_to_id = pickle.load(open('../data/protein_synonyms.pickle','rb'))
gene_protein = pickle.load(open('../data/gene_protein_ids.pickle','rb'))
network_genes = list(graph.nodes())

#%%
f = open('../data/GeneHancer_Version_4-4.csv')

enhancer_proteins = {}

header = f.readline()
for line in f:
    line = line.split(',')
    attributes = line[8]
    # print(attributes)
    # break
    connected_proteins = []
    for sp in attributes.split(';'):
        if sp.startswith('genehancer_id'):
            enhancer_id = sp.split('=')[1]

        if sp.startswith('connected_gene'):
            connected_protein = sp.split('=')[1]
            if connected_protein.startswith('EN'):
                if connected_protein in gene_protein:
                    connected_protein = gene_protein[connected_protein]
                    connected_proteins.append(connected_protein)

            elif connected_protein in protein_name_to_id:
                connected_protein = protein_name_to_id[connected_protein]
                connected_proteins += connected_protein
        
    enhancer_proteins[enhancer_id] = connected_proteins

#%%
protein_enhancers = {}
for enhancer_id in enhancer_proteins:
    for gene in enhancer_proteins[enhancer_id]:
        if gene not in protein_enhancers:
            protein_enhancers[gene] = []
        protein_enhancers[gene].append(enhancer_id)

# %%
pickle.dump(enhancer_proteins, open('../data/enhancer_proteins.pickle','wb'))
pickle.dump(protein_enhancers, open('../data/protein_enhancers.pickle','wb'))
# %%
