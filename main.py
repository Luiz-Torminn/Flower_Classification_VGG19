#%%
import os
# from early_stopping import EarlyStopping

from model import*
from utils import*
from plot_loss import*
from train import train, validate
from data_organizer import FlowerLoader
from early_stopping import EarlyStopping

import torch
from torch import nn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader


#%%
# HYPERPARAMETERS
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 5
BATCH = 10
LEARNING_RATE = 0.0001
LOAD_MODEL = True
INDEXES = {0: 'Daisy', 1: 'Rose', 2: 'Tulip', 3: 'Dandelion', 4: 'Sunflower'}

#%%
tdata = FlowerLoader(f'{os.getcwd()}/data/dataset/train')
vdata = FlowerLoader(f'{os.getcwd()}/data/dataset/val')

training_data = DataLoader(dataset = tdata, batch_size = BATCH, shuffle = True)
val_data = DataLoader(dataset = vdata, batch_size = BATCH, shuffle = False)

#%%
model = VGG19_model().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = lrs.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, cooldown = 2, threshold=0.001, verbose = True)

#%%
if LOAD_MODEL:
    try:
        model_loader(model, torch.load(f"{os.getcwd()}/Data/Saves/model_checkpoint.pth.tar"))
    except FileNotFoundError:
        print("\nNo checkpoint was found...\n")

#%%
losses = {"Train loss": [], "Test loss": []}

#%%
# Early stop implementation
es = EarlyStopping(patience=1, threshold = 1)

#%%
continue_train = True

while continue_train:
    for epoch in range(EPOCHS):
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        train_info = train(model=model, dataloader=training_data, loss_func=loss_func, optimizer=optimizer, epoch = epoch, epochs = EPOCHS, device=DEVICE)
        val_loss, val_accuracy = validate(model=model, dataloader=val_data, device = DEVICE, loss_func=loss_func)

        losses["Train loss"].append(train_info)
        losses["Test loss"].append(val_loss)

        print(f"""\nFor epoch {epoch + 1}:
        Train Loss = {train_info:.4f}
        Validation Loss = {val_loss:.4f}
        Validation Accuracy = {val_accuracy:.2f}%\n
        """)

        scheduler(val_loss).step()
        continue_train = es(model=model, val_loss=val_loss)
    
        if not continue_train:
            model_save(checkpoint, file_path = f"{os.getcwd()}/data/saves/model/model_checkpoint_{EPOCHS}_epochs_BCELoss.pth.tar")
            plot_loss(losses_dict=losses, figsave_path=f"{os.getcwd()}/data/saves/graphs")

            break


#%%
