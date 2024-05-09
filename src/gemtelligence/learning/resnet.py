from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from dataclasses import dataclass

@dataclass
class ValClass:
    loss: torch.tensor
    loss_ori: torch.tensor
    loss_heat: torch.tensor
    accu_ori: torch.tensor
    accu_ht: torch.tensor
    batchdim: int
    valid_heat: int


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=(kernel_size-1)//2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=(kernel_size-1)//2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(pl.LightningModule):
    def __init__(self, cfg, input_length, input_channels, num_classes = 6):
        super().__init__()
    
        num_layers = cfg.method.layers
        block_size = cfg.method.block_size
        hidden_num = cfg.method.hidden_size
        kernel_size = cfg.method.kernel_size
        hidden_sizes = [hidden_num] * num_layers
        num_blocks =[block_size] * num_layers
        assert len(num_blocks) == len(hidden_sizes)
        self.input_length = input_length #cfg.source.input_dim
        self.in_channels = cfg.method.in_channels
        self.input_channels = input_channels
        self.n_classes = num_classes
            
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=kernel_size, stride=1,
            padding=(kernel_size-1)//2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.dropout = nn.Dropout(cfg.method.dropout)
        self.cfg = cfg
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [self.cfg.method.stride_inner] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                stride=strides[idx],kernel_size=self.cfg.method.kernel_inner))
        self.encoder = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.z_dim, self.z_inner_dim = self._get_encoding_size()
        self.linear = nn.Linear(self.z_dim.shape[-1], self.n_classes)
        self.lr = cfg.method.lr
        # Load a pretrained model that output where the data is saturated
        if self.cfg.method.remove_saturation:
            self.base_model = self.base_model_load("BertinoPreTraining")
            self.base_model.freeze()
    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        from torch.autograd import Variable

        temp = Variable(torch.rand(5,  self.input_channels, self.input_length)) # batch, channels, length
        z_inner = self.inner_encoder(temp)
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z, z_inner

    def compute_loss(self, logits, targets):
        loss_ori = torch.tensor([float('NaN')])
        loss_heat = torch.tensor([float('NaN')])
        if self.cfg.target == "origin" or self.cfg.source.add_secondary_target:
            loss_ori = self.criterion(logits[:, :4], targets[:, 0].long())
            loss = loss_ori
        t = targets[:,1]
        if self.cfg.target == "heat" or self.cfg.source.add_secondary_target:
            indeces = t != -1
            loss_heat = self.criterion(logits[:,4:][indeces], t[indeces].long())
            loss = loss_heat
        # if self.cfg.source.add_secondary_target:
        #     if torch.isnan(loss_heat):
        #         loss = loss_ori
        #     else:
        #         loss = loss_ori + loss_heat*indeces.sum()/len(t)

        loss_ori = loss_ori
        loss = loss 
        correct_ori = torch.argmax(logits[:,:4], axis=1) == targets[:, 0]
        accu_ori = sum(correct_ori) / targets.shape[0]
        correct_ht = torch.argmax(logits[:,4:], axis=1) == targets[:, 1] 
        # if torch.isnan(loss_heat):
        #     loss_heat = torch.tensor([0])
        #     accu_ht = torch.tensor([0])
        # else:
        assert not torch.isnan(accu_ori)
        accu_ht = sum(correct_ht) / len(targets[:,1][indeces]) 
        
        return loss, loss_ori, loss_heat, accu_ori, accu_ht


        
    def configure_optimizers(self):
        if self.cfg.method.type_of_optimizer == 0:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        if self.cfg.method.patience_lr > 0:
            if self.cfg.method.target in ["val_overall","val_loss","val_ht_loss"]:
                mode = "min"
            elif self.cfg.method.target in ["val_accuracy"]:
                mode = "max"

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.3, patience=self.cfg.method.patience_lr, 
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler":scheduler, "monitor":self.cfg.method.target}
        return optimizer

    def forward(self, x):
        if self.cfg.method.remove_saturation:
            x = self.base_model(x[x==10])
        z = self.encode(x)
        z = self.dropout(z)
        if self.cfg.method.additional_relu:
            z = F.relu(z)
        return self.linear(z)

    def bottleneck(self,x):
        z = x
        return z

    def encode(self, x_raw):
        x = self.inner_encoder(x_raw)
        z = self.bottleneck(x)
        return z

    def inner_encoder(self,x_raw):
        x = x_raw
        if self.cfg.method.additional_bn:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))
        x = self.encoder(x)
        return x

    def _make_layer(self, out_channels, num_blocks, stride=1,kernel_size=None):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride,kernel_size=kernel_size))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)



def add_activation(activation='relu'):
    """
    Adds specified activation layer, choices include:
    - 'relu'
    - 'elu' (alpha)
    - 'selu'
    - 'leaky relu' (negative_slope)
    - 'sigmoid'
    - 'tanh'
    - 'softplus' (beta, threshold)
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    # SOFTPLUS DOESN'T WORK with automatic differentiation in pytorch
    elif activation == 'softplus':
        return nn.Softplus(beta=1, threshold=20)



                