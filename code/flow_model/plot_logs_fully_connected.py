# %%
import matplotlib.pyplot as plt
import json
#%%
tmstp = '20230526_093559'

# %%
logfile = open(f'../../output/{tmstp}/logs/connected_model_wildtype.log')

# %%
validation_loss = []
active_nodes = []
train_loss = []
for log in logfile:
    log = json.loads(log)
    if 'train_loss' in log:
        train_loss.append(log['train_loss'])
    elif 'validation_loss' in log:
        validation_loss.append(log['validation_loss'])


# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.plot(train_loss[:-1], label='train_loss')
axs.plot(range(0, len(train_loss)-1, 10), validation_loss[:-1], label='validation_loss')
axs.set_xlabel('Epoch')
axs.set_ylabel('Loss')
axs.legend()

#%%
# Print the best validation loss
print(min(validation_loss))
# %%
