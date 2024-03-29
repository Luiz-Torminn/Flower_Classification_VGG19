#%%
import numpy as np
import matplotlib.pyplot as plt

from model import*
from utils import*
from plot_loss import*
from data_organizer import FlowerLoader

from torch.utils.data import DataLoader
#%%
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
INDEXES = {0: 'Daisy', 1: 'Rose', 2: 'Tulip', 3: 'Dandelion', 4: 'Sunflower'}
LABEL_CHECK = True

model = VGG19_model().to(DEVICE)

#%%
try:
    model_loader(model, torch.load(f"{os.getcwd()}/data/saves/model_checkpoint_5_epochs_BCELoss.pth.tar"))
except FileNotFoundError:
    print("\nNo checkpoint was found...\n")

#%%
tstdata = FlowerLoader(f'{os.getcwd()}/data/dataset/test')
test_data = DataLoader(tstdata, batch_size=1, shuffle=True)

#%%   
i = 0

#%%
with torch.no_grad():
    img, label = next(iter(test_data))
    img, label = img.to(DEVICE), label.to(DEVICE)

    output = model(img)

    _, prediction  = output.max(1)
    
    print(f"""
      Original label: {INDEXES[label.item()]}
      Predicted label: {INDEXES[prediction.item()]}  
      """)

#%%
i += 1
img = img.squeeze().cpu()
img = np.asarray(img).transpose(1,2,0)

#%%
fig, ax = plt.subplots(figsize = (5,5))
ax.imshow(img)
ax.set_title(f'{INDEXES[prediction.item()]}')
plt.savefig(f'data/saves/prediction/predicted_{INDEXES[prediction.item()]}_{i}.png')

        

# %%
