import torch

#MODEL TRAINING
def train(model, dataloader, loss_func, optimizer, epoch, epochs, device):
    model.train()
    cumulative_loss = 0.0
    
    for i, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        prediction =  model(image)
        
        loss = loss_func(prediction, label)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        cumulative_loss += loss.item()
        
        if i%10 == 0:
            print(f"For Epoch [{epoch}/{epochs}] - Training step [{i+1}/{len(dataloader)}] --> Loss: {loss:.4f}")
    
    #Scheduler and Mean Loss
    accumulated_loss = (cumulative_loss / len(dataloader))
        
    return accumulated_loss

def validate(model, dataloader, device, loss_func):
    model.eval()
    
    num_correct = 0
    num_samples = 0
    cumulative_loss = 0.0

    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)
            scores = model(image)
            
            # accuracy
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += scores.size(0)
            
            # loss
            loss = loss_func(scores, label)
            cumulative_loss += loss.item()
    
    accumulated = cumulative_loss/len(dataloader)
    accuracy = (num_correct/num_samples)*100

    return (accumulated, accuracy)