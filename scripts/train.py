import sys
import os
# Ensure Python can find our 'src' folder
sys.path.append(os.path.abspath('.'))

from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Import the architecture we just built
from src.dataset import KeplerLightCurveDataset
from src.model import MambaCPModel

def main():
    print("Initializing training pipeline...")
    
    # 1. Load the Data
    dataset = KeplerLightCurveDataset('data/Kepler-10_Q4.parquet')
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 2. Initialize the Model
    model = MambaCPModel()
    
    # 3. Setup the PyTorch Lightning Trainer
    # We will run just 10 epochs for a quick test to verify the loss decreases
    trainer = pl.Trainer(max_epochs=10, enable_progress_bar=True, enable_model_summary=True)
    
    print("Starting training loop...")
    # 4. Train!
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
