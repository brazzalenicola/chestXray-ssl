import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = models.googlenet(pretrained=True)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # keep track of intermediate layers for CKA analysis later
        intermediate_outputs = []
        inter_x = self.encoder.maxpool2(self.encoder.conv3(
            self.encoder.conv2(self.encoder.maxpool1(self.encoder.conv1(x)))
        ))
        intermediate_outputs.append(inter_x)

        for layer in [
                      self.encoder.inception3a,
                      self.encoder.inception3b,
                      self.encoder.inception4a,
                      self.encoder.inception4b,
                      self.encoder.inception4c,
                      self.encoder.inception4d,
                      self.encoder.inception4e,
                      self.encoder.inception5a,
                      self.encoder.inception5b
                      ]:
            #every above layer is an Inception block but the object is not iterable
            inter_x = layer(inter_x)
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