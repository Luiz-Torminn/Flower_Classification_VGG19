import torch
import os

# SAVE & LOAD MODEL
def model_save(state, file_path):
  torch.save(state, file_path)

def model_loader(model, checkpoint):
   model.load_state_dict(checkpoint["state_dict"], strict=False)
   
   model.load_state_dict(checkpoint["optimizer"], strict=False)
