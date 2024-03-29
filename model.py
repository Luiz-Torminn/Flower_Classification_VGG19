from torch import nn
import torch.nn.functional as F

class VGG19_model(nn.Module):
    def __init__(self):
        super(VGG19_model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, "same")
        self.conv2 = nn.Conv2d(64,64, 3, 1, "same")
        self.conv3 = nn.Conv2d(64, 128, 3, 1, "same")
        self.conv4 = nn.Conv2d(128, 128, 3, 1, "same")
        self.conv5 = nn.Conv2d(128, 256, 3, 1, "same")
        self.conv6 = nn.Conv2d(256, 256, 3, 1, "same")
        self.conv7 = nn.Conv2d(256, 512, 3, 1, "same")
        self.conv8 = nn.Conv2d(512, 512, 3, 1, "same")
        self.max = nn.MaxPool2d(2,2)
        self.full1 = nn.Linear(512*7*7, 4096)
        self.full2 = nn.Linear(4096, 4096)
        self.full3 = nn.Linear(4096, 5)
        
    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = self.max(x)
        
        # Fourth block
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = self.max(x)
        
        # Fifth block
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = self.max(x)
        
        x = x.view(-1, 512*7*7)
        
        # Sixth block -- classifier:
        x = F.relu(self.full1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.full2(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.full3(x))
        
        return x