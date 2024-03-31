import copy

class EarlyStopping():
    def __init__(self, patience = 5, threshold = 0, load_best_model = True):
        self.patience = patience
        self.threshold = threshold
        self.load_best_model = load_best_model
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        
    def __call__(self, model, val_loss):
        if not self.best_loss:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        
        if self.best_loss - val_loss >= self.threshold:
            self.counter = 0
            self.best_loss = val_loss
            self.best_model.load_state_dict(model.state_dict())
            
        if self.best_loss - val_loss < self.threshold:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f'\nModel stopped at 5 epochs without improvement...\n')
                
                if self.load_best_model:
                    model.load_state_dict(self.best_model.state_dict())
                
                return False
        
        print(f'\nCurrent early stopping status: [{self.counter}/{self.patience}]\n')
        
        return True