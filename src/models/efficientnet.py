import torch 
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import timm

class EfficientNet(pl.LightningModule):
    def __init__(self, num_classes,  lr, weight_decay, pretrained_model = None):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        classifier = self.encoder.get_classifier()

        #for pretrained models with eg SimCLR or BarlowTwins
        if pretrained_model is not None:
            self.encoder = pretrained_model
            self.encoder.classifier = classifier

    def forward(self, x):
        # keep track of intermediate layers for CKA analysis later
        """
        intermediate_outputs = []

        inter_x = self.encoder.act1(
            self.encoder.bn1(self.encoder.conv_stem(x))
        )
        intermediate_outputs.append(inter_x)

        for layer in self.encoder.blocks:
            inter_x = layer(inter_x)
            intermediate_outputs.append(inter_x)
        
        inter_x = self.encoder.act2(
            self.encoder.bn2(self.encoder.conv_stem(inter_x))
        )
        intermediate_outputs.append(inter_x)"""
        
        x = self.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs,lbls = batch
        outputs = self.encoder(imgs)
        loss = F.binary_cross_entropy_with_logits(outputs, lbls)
        self.log("train_loss", loss)

        return loss

    def val_step(self, batch, mode):
        imgs, lbls = batch
        outputs = self.encoder(imgs)
        loss = F.binary_cross_entropy_with_logits(outputs, lbls)
        self.log(mode+"_loss", loss)

        predicted = torch.sigmoid(outputs).cpu().detach().numpy()
        #round up and down to either 1 or 0
        predicted = np.round(predicted)
        
        step_output = (predicted, lbls.cpu().detach().numpy())
        return step_output

    def val_epoch_end(self, step_otputs, mode):

        predicted = [out[0] for out in step_otputs]
        labels = [out[1] for out in step_otputs]

        #calculate how many images were correctly classified
        y_trues = np.vstack(labels)
        y_preds = np.vstack(predicted)
        val_acc =  metrics.accuracy_score(y_trues, y_preds)
        self.log(mode+"_acc", val_acc)
        
        roc_auc = metrics.roc_auc_score(y_trues,y_preds, average=None)
        chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
        for i, targets in enumerate(chexpert_targets):
            self.log(targets, roc_auc[i])
        
        print()
        print(f"Accuracy: {val_acc:.3f}")
        print(f"F1 Score: {metrics.f1_score(y_trues, y_preds, average='macro'):.3f}") 
        print("ROC-AUC score:", roc_auc)
        print()

    def validation_epoch_end(self, validation_step_otputs, mode="val"):
        self.val_epoch_end(validation_step_otputs, mode=mode)

    def validation_step(self, batch, batch_idx):
        return self.val_step(batch, mode="val")
    
    def test_epoch_end(self, outputs, mode="test"):
        self.val_epoch_end(outputs, mode=mode)

    def test_step(self, batch, batch_idx):
        return self.val_step(batch, mode="test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=(0.9, 0.999), 
            weight_decay= self.hparams.weight_decay
        )
        return optimizer
