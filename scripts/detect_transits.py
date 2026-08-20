import sys
import os
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.dataset import KeplerLightCurveDataset
from src.model import MambaCPModel

# def main():
#     # KEPLER-10
#     # dataset = KeplerLightCurveDataset('data/Kepler-10_Q4.parquet', scale_factor=1000.0)

#     # TESS
#     dataset = KeplerLightCurveDataset('data/TESS_Pi_Mensae.parquet', scale_factor=1000.0)
    
#     dataloader = DataLoader(dataset, batch_size=1)
    
#     model = MambaCPModel(d_model=64)
#     trainer = pl.Trainer(max_epochs=150, enable_progress_bar=False, logger=False)
#     print("Training stable GRU backbone...")
#     trainer.fit(model, dataloader)
    
#     model.eval()
#     with torch.no_grad():
#         x = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, :-1, :]
#         y_true = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, 1:, :]
#         predictions = model(x)
        
#         y_true_np = y_true.squeeze().numpy()
#         y_pred_np = predictions.squeeze().numpy()

#     # Empirical Conformal Prediction Math
#     residuals = y_true_np - y_pred_np
#     conformal_lower_bound = np.percentile(residuals, 3.22)
#     lower_bound_array = y_pred_np + conformal_lower_bound
    
#     anomaly_indices = np.where(y_true_np < lower_bound_array)[0]
#     print(f"Total anomaly points detected: {len(anomaly_indices)}")
    
#     clusters = []
#     if len(anomaly_indices) > 0:
#         current_cluster = [anomaly_indices[0]]
#         for i in range(1, len(anomaly_indices)):
#             if anomaly_indices[i] - anomaly_indices[i-1] <= 6:
#                 current_cluster.append(anomaly_indices[i])
#             else:
#                 clusters.append(current_cluster)
#                 current_cluster = [anomaly_indices[i]]
#         clusters.append(current_cluster)

#     print(f"Clustered into {len(clusters)} distinct planetary transit events.")

#     if len(clusters) > 1:
#         transit_centers = [np.mean(c) for c in clusters]
#         cadence_days = 29.4 / (60.0 * 24.0)
#         periods_in_days = np.diff(transit_centers) * cadence_days
#         estimated_period = np.mean(periods_in_days)
#         print(f"--> Estimated Orbital Period: {estimated_period:.4f} Earth Days")
        
#     plt.style.use('dark_background')
#     fig, ax = plt.subplots(figsize=(15, 6))
#     fig.patch.set_facecolor('#121212')
#     ax.set_facecolor('#121212')
    
#     ax.plot(lower_bound_array, color='#FF5555', linestyle='--', linewidth=1.2, label='Conformal Lower Bound (2%)')
#     ax.plot(y_pred_np, color='#00FF88', linewidth=1.0, alpha=0.8, label='Predicted Baseline')
#     ax.scatter(range(len(y_true_np)), y_true_np, color='#94A3B8', s=4, alpha=0.6, label='Nominal Flux')
    
#     if len(anomaly_indices) > 0:
#         ax.scatter(anomaly_indices, y_true_np[anomaly_indices], color='#FF0055', s=28, edgecolors='#FFFFFF', linewidths=0.8, label=f'Flagged Transits', zorder=5)

#     ax.set_title('Stable GRU-CP: Empirical Conformal Transit Detection', fontsize=14, pad=15, color='#F8F8F2')
#     ax.set_xlabel('Time Step (30-min Cadence)', color='#F8F8F2', fontsize=11)
#     ax.set_ylabel('Scaled Flux Residuals (ppt)', color='#F8F8F2', fontsize=11)
#     ax.grid(True, color='#282A36', linestyle='--', linewidth=0.5)
    
#     legend = ax.legend(loc='upper right', frameon=True, facecolor='#1E1E1E', edgecolor='#282A36')
#     for text in legend.get_texts():
#         text.set_color('#F8F8F2')
        
#     plt.tight_layout()
#     plt.savefig('detected_transits_plot.png', dpi=300, bbox_inches='tight')


def main():
    # ==========================================
    # MISSION CONTROL: Set target sensor cadence
    # Kepler = 29.4 | TESS = 2.0
    # ==========================================
    TARGET_CADENCE = 2.0  
    
    dataset = KeplerLightCurveDataset('data/TESS_Pi_Mensae.parquet', scale_factor=1000.0, cadence_minutes=TARGET_CADENCE)
    dataloader = DataLoader(dataset, batch_size=1)
    
    model = MambaCPModel(d_model=64)
    trainer = pl.Trainer(max_epochs=150, enable_progress_bar=False, logger=False)
    print("Training stable GRU backbone...")
    trainer.fit(model, dataloader)
    
    model.eval()
    with torch.no_grad():
        x = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, :-1, :]
        y_true = dataset.flux.unsqueeze(0).unsqueeze(-1)[:, 1:, :]
        predictions = model(x)
        
        
        y_true_np = y_true.squeeze().numpy()
        y_pred_np = predictions.squeeze().numpy()

    residuals = y_true_np - y_pred_np
    conformal_lower_bound = np.percentile(residuals, 3.2)
    lower_bound_array = y_pred_np + conformal_lower_bound
    
    anomaly_indices = np.where(y_true_np < lower_bound_array)[0]
    print(f"Total anomaly points detected: {len(anomaly_indices)}")
    
    # ---------------------------------------------------------
    # DYNAMIC ASTROPHYSICS METRICS 
    # ---------------------------------------------------------
    # Calculate how many data points make up a standard 3-hour transit event
    cluster_threshold = int((3 * 60) / TARGET_CADENCE)
    
    clusters = []
    if len(anomaly_indices) > 0:
        current_cluster = [anomaly_indices[0]]
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] - anomaly_indices[i-1] <= cluster_threshold:
                current_cluster.append(anomaly_indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [anomaly_indices[i]]
        clusters.append(current_cluster)

    print(f"Clustered into {len(clusters)} distinct planetary transit events.")

    if len(clusters) > 1:
        transit_centers = [np.mean(c) for c in clusters]
        cadence_days = TARGET_CADENCE / (60.0 * 24.0)
        periods_in_days = np.diff(transit_centers) * cadence_days
        estimated_period = np.mean(periods_in_days)
        print(f"--> Estimated Orbital Period: {estimated_period:.4f} Earth Days")
        
    # Visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    
    ax.plot(lower_bound_array, color='#FF5555', linestyle='--', linewidth=1.2, label='Conformal Lower Bound (3.2%)')
    ax.plot(y_pred_np, color='#00FF88', linewidth=1.0, alpha=0.8, label='Predicted Baseline')
    ax.scatter(range(len(y_true_np)), y_true_np, color='#94A3B8', s=4, alpha=0.6, label='Nominal Flux')
    
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, y_true_np[anomaly_indices], color='#FF0055', s=28, edgecolors='#FFFFFF', linewidths=0.8, label=f'Flagged Transits', zorder=5)

    ax.set_title(f'Dynamic GRU-CP: Transit Detection (Cadence: {TARGET_CADENCE}m)', fontsize=14, pad=15, color='#F8F8F2')
    ax.set_xlabel(f'Time Step ({TARGET_CADENCE}-min Cadence)', color='#F8F8F2', fontsize=11)
    ax.set_ylabel('Scaled Flux Residuals (ppt)', color='#F8F8F2', fontsize=11)
    ax.grid(True, color='#282A36', linestyle='--', linewidth=0.5)
    
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#1E1E1E', edgecolor='#282A36')
    for text in legend.get_texts():
        text.set_color('#F8F8F2')
        
    plt.tight_layout()
    plt.savefig('detected_transits_plot.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
