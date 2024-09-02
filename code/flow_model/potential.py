#%%
import torch
#%% 
# Create a network to estimate a potential function
class Potential(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
                self.layers.append(torch.nn.ReLU())
            elif i == num_layers - 1:
                self.layers.append(torch.nn.Linear(hidden_dim, 1))
            else:
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)