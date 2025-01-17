import torch
from torch import nn


class ConvLayer(torch.nn.Module):
    """Conv block used in ConvNet.
    Halves the input size"""
    def __init__(self, c_in, c_out):
        super().__init__()
        
        # First conv is element-wise transform to c_out channels
        # Then Dropout
        # Then 3x3 kernel Conv which reduces the size by 2
        # Then ReLU again
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(c_out, c_out, 3),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x


class ConvNet(torch.nn.Module):
    """ConvNet. Expects input of shape 3x3, 5x5, ..."""
    def __init__(self, input_dim=11, inp_size=3, hidden_dim=32, out_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        num_layers = inp_size // 2 # 3->1, 5->2, ... 

        # Consists of l layers that reduce size by 2 each
        # Then a final linear layer with bias to transform to out_dim
        layers = []
        for i in range(num_layers):
            layers.append(ConvLayer(c_in = input_dim if i == 0 else hidden_dim, c_out=hidden_dim))
        self.conv_layers = torch.nn.Sequential(
            *layers,
        )
        self.final_layer = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.hidden_dim)
        return self.final_layer(x)


import timm


class PartsOfResNet18(torch.nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        resnet18 = timm.create_model('resnet18', pretrained=True)
        #self.layers = torch.nn.Sequential(*[
        #    torch.nn.Conv2d(11, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        #    resnet18.bn1,
        #    resnet18.act1,
        #    resnet18.maxpool,
        #    resnet18.layer1,
        #    resnet18.layer2,
        #    resnet18.global_pool,
        #    torch.nn.Linear(128, out_dim)
        #])
        self.layers=resnet18
        self.layers.conv1 = torch.nn.Conv2d(11, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = torch.nn.Linear(512, out_dim)
    
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    porene = PartsOfResNet18()
    inp = torch.randn(4096, 11, 32, 32)
    print(porene(inp).shape)
