#%%
import matplotlib.pyplot as plt
import os

#%%
def plot_loss(losses_dict, figsave_path = None):
    fig, ax = plt.subplots(figsize = (10,5))
    
    for k, v in losses_dict.items():
        ax.plot([i+1 for i in range(len(v))], v, label = f"{k.title()}")
    
    plt.suptitle("Loss x Epochs")
    plt.legend()
    
    plt.imshow()
    
    if figsave_path:
        plt.savefig(figsave_path)

# %%
