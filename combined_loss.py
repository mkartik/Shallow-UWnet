from VGG_loss import *
from torchvision import models

class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config['device'])
        self.l1loss = nn.L1Loss().to(config['device'])

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        total_loss = mse_loss + vgg_loss
        return total_loss, mse_loss, vgg_loss