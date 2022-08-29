import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Densenet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = models.densenet121(pretrained=True)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # keep track of intermediate layers for CKA analysis later
        intermediate_outputs = []
        inter_x = self.encoder.features.pool0(self.encoder.features.relu0(
            self.encoder.features.norm0(self.encoder.features.conv0(x))
        ))
        intermediate_outputs.append(inter_x)

        for block in [
                      self.encoder.features.denseblock1,
                      self.encoder.features.transition1,
                      self.encoder.features.denseblock2,
                      self.encoder.features.transition2,
                      self.encoder.features.denseblock3,
                      self.encoder.features.transition3,
                      self.encoder.features.denseblock4
                      ]:
            #for layer in block:
            inter_x = block(inter_x)
            intermediate_outputs.append(inter_x)
        intermediate_outputs.append(self.encoder.features.norm5(inter_x))
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