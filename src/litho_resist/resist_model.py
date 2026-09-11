import torch
import torch.nn as nn

class ResistUNet(nn.Module):
    def __init__(self):
        super(ResistUNet, self).__init__()
        # Clean modular PyTorch code here
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.out(self.relu(self.conv1(x))))
