import torch
import torch.nn as nn
import pytorch_lightning as pl

class MambaCPModel(pl.LightningModule):
    def __init__(self, d_model=64):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Linear(1, d_model)
        
        # Reverting to the stable GRU (its internal gates prevent signal explosion)
        self.sequence_mixer = nn.GRU(d_model, d_model, batch_first=True) 
        
        self.decoder = nn.Linear(d_model, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.sequence_mixer(x)
        predictions = self.decoder(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1, :] 
        y = batch[:, 1:, :]  
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
