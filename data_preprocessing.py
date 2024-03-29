#%%
import os 
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# %%
def create_folders():
    FOLDERS = ['train', 'val', 'test']
    
    for folder in FOLDERS:
        os.mkdir(f"data/dataset/{folder}") if not os.path.exists(f"data/dataset/{folder}") else print(f"folder {folder} already exists") 
    
create_folders()
# %%
#Divide files into their respective dataset folder
def images_division(folder_path, dataset_path):
    FOLDERS = os.listdir(folder_path)
    DATASET = ['train', 'val', 'test']
    
    for folder in FOLDERS:
        if folder == ".DS_Store":
                continue
            
        files = os.listdir(f'{folder_path}/{folder}')
        files = [i for i in files if i != ".DS_Store"]
        train_ds, val_ds = train_test_split(files, train_size=0.85, test_size=0.15)
        train_ds, tst_ds = train_test_split(train_ds, train_size=0.99, test_size=0.01)

        for dataset in DATASET:
            try:
                if dataset == "train":
                    [os.rename(f'{folder_path}/{folder}/{file}', f'{dataset_path}/{dataset}/({folder})_{dataset}_{i+1}.jpg') for i, file in enumerate(train_ds)] 
                
                if dataset == "val":
                    [os.rename(f'{folder_path}/{folder}/{file}', f'{dataset_path}/{dataset}/({folder})_{dataset}_{i+1}.jpg') for i, file in enumerate(val_ds)] 
                
                if dataset == "test":
                    [os.rename(f'{folder_path}/{folder}/{file}', f'{dataset_path}/{dataset}/({folder})_{dataset}_{i+1}.jpg') for i, file in enumerate(tst_ds)] 
            except FileNotFoundError:
                pass
        
images_division('flowers', 'data/dataset')        
# %%
