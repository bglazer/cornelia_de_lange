# %%
import pickle
graph = pickle.load(open('../data/filtered_graph.pickle','rb'))
protein_name_to_id = pickle.load(open('../data/protein_synonyms.pickle','rb'))
protein_id_to_name = pickle.load(open('../data/protein_id_to_name.pickle','rb'))
gene_protein = pickle.load(open('../data/gene_protein_ids.pickle','rb'))
network_proteins = list(graph.nodes())

#%%
f = open('../data/GeneHancer_Version_4-4.csv')

enhancer_proteins = {}
enhancer_coords = {}

header = f.readline()
for line in f:
    line = line.split(',')
    attributes = line[8]
    # print(attributes)
    # break
    connected_proteins = []
    chr = line[0]
    start = int(line[3])
    end = int(line[4])

    for sp in attributes.split(';'):
        if sp.startswith('genehancer_id'):
            enhancer_id = sp.split('=')[1]
            enhancer_coords[enhancer_id] = (chr, start, end)
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
    for protein in enhancer_proteins[enhancer_id]:
        if protein not in protein_enhancers:
            protein_enhancers[protein] = []
        protein_enhancers[protein].append(enhancer_id)

# %%
pickle.dump(enhancer_proteins, open('../data/enhancer_proteins.pickle','wb'))
pickle.dump(protein_enhancers, open('../data/protein_enhancers.pickle','wb'))
# %%
# Count the number of enhancers that are connected to each protein in the Nanog regulatory network
n_enhancers = []
from collections import Counter
enhancer_counter = Counter()
for protein in network_proteins:
    if protein in protein_enhancers:
        n = len(protein_enhancers[protein])
    else:
        n = 0
    print(f'{protein} {protein_id_to_name[protein]}: {n}')
    n_enhancers.append(n)
    enhancer_counter[n] += 1

for n, count in enhancer_counter.most_common():
    print(f'{n}: {count}')
    
from matplotlib import pyplot as plt
plt.hist(n_enhancers, bins=20);
# %%
# Count the number of genes in the Nanog network that are shared across enhancers
enhancer_network_proteins = {}
for enhancer in enhancer_proteins:
    for protein in enhancer_proteins[enhancer]:
        if protein in network_proteins:
            if enhancer not in enhancer_network_proteins:
                enhancer_network_proteins[enhancer] = []
            enhancer_network_proteins[enhancer].append(protein)
pickle.dump(enhancer_network_proteins, open('../data/enhancer_network_proteins.pickle','wb'))
#%%
enhancer_network_gene_counts = {enhancer: len(genes) for enhancer, genes 
                                in enhancer_network_proteins.items()}

for n, count in Counter(enhancer_network_gene_counts.values()).most_common():
    print(f'{n}: {count}')


#%%
# Invert the enhancer coordinates
coords_enhancers = {}
for enhancer in enhancer_coords:
    chr, start, end = enhancer_coords[enhancer]
    if chr not in coords_enhancers:
        coords_enhancers[chr] = []
    coords_enhancers[chr].append((start, end, enhancer))

# %%
# Load TAD annotations
import os

tad_dir = '../data/TAD_annotations/TADs'
tad_files = os.listdir(tad_dir)

tad_regions = {}
for f in tad_files:
    for line in open(os.path.join(tad_dir, f)):           
        chromosome, start, end = line.strip().split(' ')
        if chromosome not in tad_regions:
            tad_regions[chromosome] = []
        tad_regions[chromosome].append((int(start),int(end)))

# sort tad starts along with the index to the end
for chromosome in tad_regions:
    tad_regions[chromosome] = sorted(tad_regions[chromosome])

#%%
from collections import defaultdict
tad_enhancer_linkages = defaultdict(list)

for chromosome in tad_regions:
    tad_idx = 0
    enh_idx = 0
    tads = tad_regions[chromosome]
    enhancers = coords_enhancers[chromosome]
        
    while True:
        tad_start, tad_end = tads[tad_idx]
        enh_start, enh_end, enh = enhancers[enh_idx]
        tad_term = f'tad-{chromosome}-{tad_start}-{tad_end}'
        if (enh_start <= tad_end and enh_end >= tad_start):
            tad_enhancer_linkages[tad_term].append(enh)
            enh_idx = enh_idx+1
            if enh_idx==len(enhancers):
                break
        elif enh_start <= tad_start:
            enh_idx = enh_idx+1
            if enh_idx==len(enhancers):
                break
        elif enh_start >= tad_end:
            tad_idx = tad_idx+1
            if tad_idx == len(tads):
                break
tad_enhancer_linkages = dict(tad_enhancer_linkages)
# %%
tad_enhancer_genes = {}

for tad in tad_enhancer_linkages:
    for enhancer in tad_enhancer_linkages[tad]:
        if enhancer in enhancer_network_proteins:
            if tad not in tad_enhancer_genes:
                tad_enhancer_genes[tad] = set()
            tad_enhancer_genes[tad].update(enhancer_network_proteins[enhancer])
# %%
# Count the number of genes in the Nanog network that are shared across TADs
tad_network_gene_counts = {tad: len(genes) for tad, genes 
                           in tad_enhancer_genes.items()}

for n, count in Counter(tad_network_gene_counts.values()).most_common():
    print(f'{n}: {count}')

# Number of unique genes associated with TAD enhancers
print('Number of unique genes associated with TAD enhancers')
print(len(set([gene for tad in tad_enhancer_genes for gene in tad_enhancer_genes[tad]])))

# %%
