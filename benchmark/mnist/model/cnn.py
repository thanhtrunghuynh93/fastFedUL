from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
import torch

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)
        ## apply norm layer
        self.norm1 = nn.LayerNorm([32, 28, 28])
        self.norm2 = nn.LayerNorm([64, 14, 14])
        self.norm_fc = nn.LayerNorm(512)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            # m.weight.data.normal_(mean=0.0, std=1.0)
            # if m.bias is not None:
            #     m.bias.data.zero_()
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.norm_fc(self.fc1(x)))
        return x

    def decoder(self, x):
        x = self.fc2(x)
        return x
    
    def pred_and_rep(self, x):
        e = self.encoder(x)
        o = self.decoder(e)
        return o, e

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)