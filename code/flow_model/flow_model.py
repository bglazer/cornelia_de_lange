from torch.nn import Linear, LeakyReLU, ReLU
from torch import FloatTensor
import torch
import networkx as nx
from networkx.generators import ego_graph
from tqdm import tqdm

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, input_bias=True):
        super(MLP, self).__init__()
        layers = []
        layers.append(Linear(input_dim, hidden_dim, bias=input_bias))
        layers.append(ReLU())
        for i in range(num_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(ReLU())
            # TODO do we need batch norm here?
        layers.append(Linear(hidden_dim, output_dim, bias=False))
        # layers.append(LeakyReLU())
        # Register the layers as a module of the model
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class GroupL1MLP(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(GroupL1MLP, self).__init__(input_dim, output_dim, hidden_dim, num_layers, False)

        self.group_l1 = torch.ones(self.layers[0].weight.shape[1])
        self.group_l1 = torch.nn.Parameter(self.group_l1, requires_grad=True)

    def forward(self, x):
        x = x @ (self.layers[0].weight * torch.relu(self.group_l1)).T

        for layer in self.layers[1:]:
            x = layer(x)
        return x

class LinearFlowModel(torch.nn.Module):
    def __init__(self, input_dim, predict_var=False):
        super(LinearFlowModel, self).__init__()
        self.models = []
        self.predict_var = predict_var
        outdim = 2 if predict_var else 1
        # Create a separate Linear model for each node in the graph
        for i in range(input_dim):
            node = Linear(input_dim, outdim, bias=True)
            self.models.append(node)
        self.models = torch.nn.ModuleList(self.models)
        

    def forward(self, state):
        delta = torch.zeros_like(state)
        if self.predict_var:
            var = torch.zeros_like(state)
        # For each node, run the corresponding MLP and insert the output into the delta tensor
        for i,model in enumerate(self.models):
            # state is a tensor of shape (num_nodes, num_states)
            y = model(state)
            if self.predict_var:
                delta[:,i] = y[:,0]
                var[:,i] = y[:,1]
            # Add the weights of the first layer to a list
        # Get the weights of the first layer
        if self.predict_var:
            return delta, var
        return delta

# Variational Autoencoder (VAE)
class VAEFlowModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(VAEFlowModel, self).__init__()
        self.encoder = GroupL1MLP(input_dim, latent_dim*2, hidden_dim, num_layers)
        self.decoder = MLP(latent_dim, 1, hidden_dim, num_layers)

    def forward(self, x, nsamples=1):
        encoded = self.encoder(x)
        halfdim = int(encoded.shape[1]/2)
        mu = encoded[:, :halfdim]
        logvar = encoded[:, halfdim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn(nsamples, std.shape[0], std.shape[1], device=std.device)
        z = mu + eps*std
        decoded = self.decoder(z)
        return decoded
    
class ConnectedFlowModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, predict_var=False):
        super(ConnectedFlowModel, self).__init__()
        self.output_dim = output_dim
        self.predict_var = predict_var
        if predict_var:
            self.model = MLP(input_dim, output_dim*2, hidden_dim, num_layers)
        else:
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers)

    def forward(self, state):
        # state is a tensor of shape (num_nodes, num_states)
        delta = self.model(state)[:, :self.output_dim]
        if self.predict_var:
            var = self.model(state)[:, self.output_dim:]
            return delta, var
        return delta
    
class L1FlowModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(L1FlowModel, self).__init__()
        self.models = []
        # Create a separate MLP for each node in the graph
        for i in range(input_dim):
            node = MLP(input_dim, 1, hidden_dim, num_layers, input_bias=False)
            self.models.append(node)
        self.models = torch.nn.ModuleList(self.models)

    def forward(self, state):
        delta = torch.zeros_like(state)
        # For each node, run the corresponding MLP and insert the output into the delta tensor
        for i,model in enumerate(self.models):
            # state is a tensor of shape (num_nodes, num_states)
            delta[:,i] = model(state).squeeze()
            # Add the weights of the first layer to a list
        # Get the weights of the first layer
        return delta
    
class GroupL1FlowModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, predict_var=False):
        super(GroupL1FlowModel, self).__init__()
        self.models = []
        self.predict_var = predict_var
        outdim = 2 if predict_var else 1
        # Create a separate MLP for each node in the graph
        for i in range(input_dim):
            node = GroupL1MLP(input_dim, outdim, hidden_dim, num_layers)
            self.models.append(node)
        self.models = torch.nn.ModuleList(self.models)
        

    def forward(self, state):
        delta = torch.zeros_like(state)
        if self.predict_var:
            var = torch.zeros_like(state)
        # For each node, run the corresponding MLP and insert the output into the delta tensor
        for i,model in enumerate(self.models):
            # state is a tensor of shape (num_nodes, num_states)
            y = model(state)
            if self.predict_var:
                delta[:,i] = y[:,0]
                var[:,i] = y[:,1]
            # Add the weights of the first layer to a list
        # Get the weights of the first layer
        if self.predict_var:
            return delta, var
        return delta

class GraphFlowModel(torch.nn.Module):
    def __init__(self, num_layers, graph, data_idxs, hops=1):
        super(GraphFlowModel, self).__init__()
        # Make an MLP for each node in the networkx graph
        # with one input to the MLP for each incoming edge and a 
        self.graph = graph
        self.data_idxs = data_idxs
        self.models = {}
        # self.models = torch.nn.ModuleList(self.models)

        # Remove nodes that are not in the input data
        non_data_nodes = []
        for node in graph.nodes():
            if node not in data_idxs:
                print(f'Error: node {node} is in the graph but it is not in the input data')
                non_data_nodes.append(node)
        graph.remove_nodes_from(non_data_nodes)

        # Remove nodes that have no incoming edges
        removed = True
        while removed:
            zero_indegree_nodes = []
            removed = False
            for node, indegree in graph.in_degree():
                if indegree == 0:
                    print(f'Error: node {node} has no incoming edges')
                    zero_indegree_nodes.append(node)
                    removed = True
            graph.remove_nodes_from(zero_indegree_nodes)

        # Add self loops to the graph
        for node in graph.nodes():
            if not graph.has_edge(node, node):
                graph.add_edge(node, node)
        
        # Convert the graph into a list of indexes
        self.adj_list = {}
        non_data_nodes = []
        for node in self.graph.nodes():
            self.adj_list[node] = []
            for neighbor in ego_graph(self.graph, node, radius=hops):
                if neighbor in self.data_idxs:
                    self.adj_list[node].append(data_idxs[neighbor])
                else:
                    raise Exception(f'Error: node {neighbor} is in the graph but it is not in the input data')
        
        for node in self.adj_list:
            input_dim = len(self.adj_list[node])
            # TODO make this a parameter
            # minimum hidden dimension is 10
            hidden_dim = max(10, input_dim*2)
            model = MLP(input_dim, 1, hidden_dim, num_layers)
            self.models[node] = model

        self.models = torch.nn.ModuleDict(self.models)
        
    def forward(self, state, node):
        # Apply each model to the portion of the state vector that
        #  corresponds to the inputs to the node in the graph
        inputs = state[:,self.adj_list[node]]
        delta = self.models[node](inputs)

        return delta[:,0]

class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        num_nodes = len(models)
        num_models = len(models[0])
        self.num_nodes = num_nodes
        self.num_models = num_models
        # Get the size of the hidden layer from the first model
        # Assumes that all models have the same hidden layer size
        hidden_dim = models[0][0]['layers.0.weight'].shape[0]
        num_layers = len([key for key in models[0][0] if 'weight' in key])-1
        print('Initializing ensemble')
        self.models = []
        self.model_gpu = {}
        for model_idx in tqdm(range(num_models)):
            new_model = GroupL1FlowModel(input_dim=num_nodes, 
                                         hidden_dim=hidden_dim, 
                                         num_layers=num_layers)
            self.models.append(new_model)
        print('Loading weights')
        for node_idx, node_ensemble in tqdm(enumerate(models)):
            for model_idx, weights in enumerate(node_ensemble):
                m = self.models[model_idx].models[node_idx]
                m.load_state_dict(weights)
                
            self.models[model_idx].eval()

    def forward(self, x, mean=True):
        with torch.no_grad():
            y = torch.zeros(self.num_models, x.shape[0], self.num_nodes)
            for model_idx, model in enumerate(self.models):
                y[model_idx] = model(x)
            if mean:
                return y.mean(dim=0)
            else:
                return y