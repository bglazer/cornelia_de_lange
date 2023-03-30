from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
from torchdyn.core import NeuralODE

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
def MLP(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(LeakyReLU())
    for i in range(num_layers - 1):
        layers.append(Linear(hidden_dim, hidden_dim))
        layers.append(LeakyReLU())
        # TODO do we need batch norm here?
    layers.append(Linear(hidden_dim, output_dim, bias=False))
    layers.append(LeakyReLU())
    return Sequential(*layers)

# FlowModel is a neural ODE that takes in a state and outputs a delta
class FlowModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(FlowModel, self).__init__()
        self.model = MLP(input_dim, output_dim, hidden_dim, num_layers)
        # self.neural_ode = NeuralODE(self.model, sensitivity='adjoint')  

    def forward(self, state, tspan):
        # state is a tensor of shape (num_nodes, num_states)
        delta = self.model(state)
        return delta

    # def trajectory(self, state, tspan):
    #     # state is a tensor of shape (num_nodes, num_states)
    #     # tspan is a tensor of shape (num_timesteps,)
    #     trajectory = self.neural_ode.trajectory(state, tspan)
    #     return trajectory