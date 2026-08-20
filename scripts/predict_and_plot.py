import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.dataset import KeplerLightCurveDataset
from src.model import MambaCPModel

def main():
    dataset = KeplerLightCurveDataset('data/Kepler-10_Q4.parquet', scale_factor=1000.0)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Train model on zero-centered ppt data
    model = MambaCPModel(d_model=64)
    trainer = pl.Trainer(max_epochs=150, enable_progress_bar=False, logger=False)
    print("Training Mamba-CP on scaled photometric variations (150 epochs)...")
    trainer.fit(model, dataloader)
    
    model.eval()
    with torch.no_grad():
        x = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, :-1, :] 
        y_true = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, 1:, :]
        
        predictions = model(x)
        
        # Invert scaling back to standard normalized flux: flux = (scaled / 1000.0) + 1.0
        scale = dataset.scale_factor
        y_true_np = (y_true.squeeze().numpy() / scale) + 1.0
        lower_bound = (predictions[0, :, 0].numpy() / scale) + 1.0
        median_pred = (predictions[0, :, 1].numpy() / scale) + 1.0
        upper_bound = (predictions[0, :, 2].numpy() / scale) + 1.0

    print("Generating refined visualization...")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    
    # Plot Conformal Prediction Envelope
    ax.fill_between(range(len(y_true_np)), lower_bound, upper_bound, color='#00D2FF', alpha=0.35, label='90% Conformal Confidence Band')
    
    # Plot predicted median
    ax.plot(median_pred, color='#00FF88', linewidth=1.2, label='Predicted Stellar Trend (Median)')
    
    # Plot raw sensor data
    ax.scatter(range(len(y_true_np)), y_true_np, color='#FF5577', s=4, label='Actual Sensor Flux', zorder=3)
    
    ax.set_title('Mamba-CP: High-Precision Conformal Envelope Tracking', fontsize=14, pad=15, color='#F8F8F2')
    ax.set_xlabel('Time Step (30-min Cadence)', color='#F8F8F2', fontsize=11)
    ax.set_ylabel('Normalized Flux', color='#F8F8F2', fontsize=11)
    ax.grid(True, color='#282A36', linestyle='--', linewidth=0.5)
    
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#1E1E1E', edgecolor='#282A36')
    for text in legend.get_texts():
        text.set_color('#F8F8F2')
        
    plt.tight_layout()
    plt.savefig('conformal_bounds_plot.png', dpi=300, bbox_inches='tight')
    print("Updated conformal_bounds_plot.png with scaled tracking.")

if __name__ == "__main__":
    main()
