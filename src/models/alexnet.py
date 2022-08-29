import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = models.alexnet(pretrained=True)
        self.encoder.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        # keep track of intermediate layers for CKA analysis later
        intermediate_outputs = []

        inter_x = x 
        for layer in self.encoder.features:
            inter_x = layer(inter_x)
            intermediate_outputs.append(inter_x)
        
        inter_x = self.encoder.avgpool(inter_x)
        intermediate_outputs.append(inter_x)
            
        x = self.encoder(x)
        return x, intermediate_outputs

    def configure_optimizer(self, lr, weight_decay=0.0):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=lr, 
            betas=(0.9, 0.999), 
            weight_decay=weight_decay
        )
        scheduler = None # TODO
        return optimizer, scheduler