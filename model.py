import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=61, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)

    def forward(self, x):
        x, input_x = x
        a = self.relu(self.conv1(self.relu(self.drop(self.conv(self.relu(self.drop(self.conv(x))))))))
        out = torch.cat((a, input_x), 1)
        return (out, input_x)

class UWnet(nn.Module):
    def __init__(self, num_layers=3):
        super(UWnet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = self.StackBlock(ConvBlock, num_layers)

    def StackBlock(self, block, layer_num):
        layers = []
        for _ in range(layer_num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        input_x = x
        x1 = self.relu(self.input(x))
        out, _ = self.blocks((x1, input_x))
        out = self.output(out)
        return out

