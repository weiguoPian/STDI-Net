import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DI_net(nn.Module):
    def __init__(self, args):
        super(DI_net, self).__init__()
        self.args = args
        self.input_dim = args.hour_dim
        self.mod_weight = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=512),
            #nn.Linear(in_features=self.input_dim, out_features=1024),
            nn.LeakyReLU()
        )
        self.mod_bias = nn.Sequential(
            nn.Linear(
                in_features=self.input_dim, out_features=self.args.NYC_Height*self.args.NYC_Weight*2),
            #nn.ReLU()
            nn.LeakyReLU()
        )

        self.left = nn.Parameter(torch.Tensor(1024, 512))
        self.right = nn.Parameter(torch.Tensor(512, self.args.NYC_Height*self.args.NYC_Weight*2))
        nn.init.kaiming_normal_(self.left)
        nn.init.kaiming_normal_(self.right)

    def forward(self, x):
        weight_res = self.mod_weight(x)
        weight_out = []
        for i in range(len(weight_res)):
            mid = torch.mm(torch.mm(self.left, torch.diag(weight_res[i])), self.right)
            weight_out.append(mid)
        weight_out = torch.stack(weight_out)
        weight_out = weight_out.view(-1, self.args.NYC_Height*self.args.NYC_Weight*2, 1024)

        bias_out = self.mod_bias(x)
        return weight_out, bias_out

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.conv1 = self.convlayer(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = self.convlayer(in_channels=out_channels, out_channels=out_channels)
    
    def convlayer(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=(3,3),
                         stride=1,
                         padding=1,
                         bias=True)
        nn.init.kaiming_normal_(conv.weight)

        out = nn.Sequential(conv,
                            nn.BatchNorm2d(num_features=out_channels))
        return out
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = out1 + out2
        out = F.relu(out)
        return out

class STDI_Net(nn.Module):
    def __init__(self, args):
        super(STDI_Net, self).__init__()
        self.args = args

        self.spatial = nn.ModuleList()
        for i in range(self.args.seq_len):
            self.spatial.append(self.ConvBlock(in_channels=2, out_channels=32))

        self.seq_lstm = nn.LSTM(
            input_size=self.args.NYC_Height*self.args.NYC_Weight*32, hidden_size=1024, num_layers=1, batch_first=False)

        self.di = DI_net(self.args)
    
    def forward(self, x, h):
        di_weight, di_bias = self.di(h)
        spatial_out = []
        for i in range(self.args.seq_len):
            seq_item = self.spatial[i](x[i])
            seq_item = seq_item.view(seq_item.size(0), -1)
            spatial_out.append(seq_item)
        spatial_out = torch.stack(spatial_out)
        lstm_out, (hn, cn) = self.seq_lstm(spatial_out)
        lstm_out = lstm_out[-1]
        out = torch.sum(lstm_out.unsqueeze(1)*di_weight, dim=2) + di_bias
        out = F.relu(out)
        return out

    def ConvBlock(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels=out_channels,
                         out_channels=out_channels,
                         kernel_size=(3,3),
                         stride=1,
                         padding=1,
                         bias=True)
        nn.init.kaiming_normal_(conv.weight)
        block = nn.Sequential(ResUnit(in_channels=in_channels,out_channels=out_channels),
                              ResUnit(in_channels=out_channels,out_channels=out_channels),
                              conv,
                              nn.BatchNorm2d(num_features=out_channels),
                              nn.ReLU())
        return block