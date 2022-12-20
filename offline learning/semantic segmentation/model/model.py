import torch
import torch.utils as utils
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

class DTSegmentationNetwork(pl.LightningModule):
    """CNN for semantic segmentation."""
    def __init__(self, hparams):
        """
        Hyperparameters required:
            # Model
            - num_classes: number of classes to be segmented
            
            # Optimizer
            - learning_rate: learning rate for the optimizer
            - lr_decay: learning rate decay for the optimizer
            - weight_decay: weight decay for the optimizer
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        # Load the pretrained model and disable gradient computation for all layers
        self.model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the last segmentation layers with new 
        # more complex ones to increase model's capacity
        # self.model.classifier.cbr = torch.nn.Sequential(
        #     torch.nn.Conv2d(960, 580, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     torch.nn.BatchNorm2d(580, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
            
        #     torch.nn.Conv2d(580, 232, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     torch.nn.BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True),
            
        #     torch.nn.Conv2d(232, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     torch.nn.ReLU(inplace=True)
        # )
        
        # Enable gradient computation for the last segmentation layer to be fine-tuned
        self.model.classifier.requires_grad_(True)
        self.model.classifier.low_classifier = torch.nn.Conv2d(in_channels=40, out_channels=hparams['num_classes'], kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier.high_classifier = torch.nn.Conv2d(in_channels=128, out_channels=hparams['num_classes'], kernel_size=(1, 1), stride=(1, 1))
        # Integrate softmax layer into the model to avoid having to apply it manually 
        self.model.classifier.add_module("softmax", torch.nn.Softmax())

    def forward(self, x):
        return self.model(x)["out"]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        print(f'x: {x.shape}, y: {y.shape}, y_hat: {y_hat.shape}')
        
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        print(f'x: {x.shape}, y: {y.shape}, y_hat: {y_hat.shape}')
        
        val_loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.hparams['learning_rate'], weight_decay=self.hparams['weight_decay'])
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=self.hparams['lr_decay'], verbose=True), 
            'monitor': 'val_loss'
        }
        
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler}
    
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print(f'Saving model... {path}')
        torch.save(self, path)


if __name__ == '__main__':
    model = DTSegmentationNetwork(hparams={'num_classes': 5, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'lr_decay': 0.1})
    print(model)
