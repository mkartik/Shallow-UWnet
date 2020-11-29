from __future__ import print_function
from __future__ import division

import torch

from model import UWnet
from ptflops import get_model_complexity_info
from flopth import flopth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UWnet(num_layers = 3).to(device)

flops, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True)
print('Densenet: {:<30}  {:<8}'.format('Computational complexity: ', flops))
print('Densenet: {:<30}  {:<8}'.format('Number of parameters: ', params))
pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total - ', pytorch_total_params)
print('Trainable - ',trainable_pytorch_total_params)

from pytorch_model_summary import summary
print(summary(UWnet(num_layers = 3), torch.zeros((64,3,3,3)), show_input=False, show_hierarchical=False))
