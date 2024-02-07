#%%
import pickle
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
from flow_model import GroupL1FlowModel
import sys
sys.path.append('..')
from util import embed_velocity, velocity_vectors
import scanpy as sc
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
#%%
genotype = 'wildtype'
tmstp = '20230607_165324'
adata = sc.read_h5ad(f'../../data/{genotype}_net.h5ad')

#%%
losses = {}
active_inputs = {}
l1_alphas = {}

logfiles = glob.glob(f'../../output/{tmstp}/logs/*.log')
for path in logfiles:
    filename = path.split('/')[-1]
    node = int(filename.split('_')[3])
    with open(path, 'r') as f:
        for line in f:
            log = json.loads(line)
            if 'epoch' not in log:
                if node not in losses:
                    losses[node] = []
                    active_inputs[node] = []
                    l1_alphas[node] = []
                losses[node].append(log['val_mean_loss'])
                active_inputs[node].append(log['active_inputs'])
                l1_alphas[node].append(log['l1_alpha'])
# %%
