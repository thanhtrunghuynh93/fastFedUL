import torch
from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F

# class Model(FModule):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 64, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1600, 384),
#             nn.ReLU(),
#             nn.Linear(384, 192),
#             nn.ReLU(),
#             nn.Linear(192, 10),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.flatten(1)
#         return self.decoder(x)
    
# l1 = 64
# l2 = 128
# l3 = 256
# class Model(FModule):
#     def __init__(self):
#         super(Model, self).__init__()

#         # input channel, output filter, kernel size
#         self.conv1 = nn.Conv2d(3, l1, 5, padding=2)
#         self.conv2 = nn.Conv2d(l1, l2, 5, padding=2)
#         self.conv3 = nn.Conv2d(l2, l3, 3)

#         self.BatchNorm2d1 = nn.BatchNorm2d(l1)
#         self.BatchNorm2d2 = nn.BatchNorm2d(l2)
#         self.BatchNorm2d3 = nn.BatchNorm2d(l3)

#         self.BatchNorm2d4 = nn.BatchNorm2d(120)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(p=0)

#         self.fc1 = nn.Linear(l3 * 3 * 3, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.BatchNorm2d1(self.pool(F.leaky_relu(self.conv1(x))))
#         x = self.BatchNorm2d2(self.pool(F.leaky_relu(self.conv2(x))))
#         x = self.BatchNorm2d3(self.pool(F.leaky_relu(self.conv3(x))))
#         x = x.view(-1, l3 * 3 * 3)
#         x = self.dropout(F.leaky_relu(self.fc1(x)))
#         x = self.dropout(F.leaky_relu(self.fc2(x)))
#         #x = self.dropout(F.leaky_relu(self.fc4(x)))
#         x = self.fc3(x)
#         return x

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Testing
        self.norm1 = nn.LayerNorm([64, 32, 32])
        self.norm2 = nn.LayerNorm([64, 16, 16])
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
        # x = x.view((x.shape[0],28,28))
        # x = x.unsqueeze(1)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Testing ideas
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
   
# class Model(FModule):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         )
#         self.decoder = nn.Linear(in_features=512, out_features=10, bias=True)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.flatten(1)
#         return self.decoder(x) 

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)