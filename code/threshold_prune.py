import torch
from copy import deepcopy
from torch.nn.utils import prune

def loss_threshold_l1_prune(model, points, V, loss_threshold, resolution=10):
    mse = torch.nn.MSELoss()
    pV, original_weights = model(points)
    original_loss = mse(pV, V).item()
    print(f'Original loss: {original_loss:.9f}')

    with torch.no_grad():
        losses = []
        nonzero_counts = []
        prune_pcts = torch.linspace(0, 1, resolution)

        for pct in prune_pcts:
            # Make a copy of the original model
            pruned_model = deepcopy(model)
            # Prune the first layer of the model
            prune.l1_unstructured(pruned_model.model[0], name='weight', amount=float(pct))
            num_nonzero_weights = torch.sum(torch.abs(pruned_model.model[0].weight) > 0)
            pV, input_weights = pruned_model(points)
            # Compute the loss between the predicted and true velocity vectors
            loss = mse(pV, V)
            losses.append(loss.item())
            nonzero_counts.append(num_nonzero_weights.item())
    
    # Find the first index where the loss is less than the threshold
    idx = torch.argmin(torch.abs(torch.tensor(losses) - original_loss*(1.0+loss_threshold)))
    # Prune the model to that percentage
    prune.l1_unstructured(model.model[0], name='weight', amount=float(prune_pcts[idx]))

    # Return the pruned model and the percentage of weights pruned
    # as well as the number of nonzero weights and the losses
    return model, idx, prune_pcts, nonzero_counts, losses
