import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # keep track of intermediate layers for CKA analysis later
        intermediate_outputs = []
        inter_x = self.encoder.maxpool(self.encoder.relu(
            self.encoder.bn1(self.encoder.conv1(x))
        ))
        intermediate_outputs.append(inter_x)

        for layer in [
                      self.encoder.layer1,
                      self.encoder.layer2,
                      self.encoder.layer3,
                      self.encoder.layer4
                      ]:
            for res_block in layer:
                inter_x = res_block(inter_x)
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