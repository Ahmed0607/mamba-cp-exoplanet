import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class KeplerLightCurveDataset(Dataset):
    # def __init__(self, parquet_file, scale_factor=1000.0, filter_window=51):
    #     self.df = pd.read_parquet(parquet_file)
        
    #     # 1. Extract the raw continuous flux
    #     raw_flux = self.df['flux'].values
        
    #     # 2. Apply a digital moving-median filter to isolate the low-frequency stellar wave
    #     # A window of 51 steps (~25 hours) perfectly captures the star's rotation while ignoring quick transits
    #     wave_baseline = self.df['flux'].rolling(window=filter_window, center=True, min_periods=1).median().values
        
    #     # 3. Flatten the signal by subtracting the wave, then scale to ppt
    #     flattened_flux = (raw_flux - wave_baseline) * scale_factor
        
    #     self.flux = torch.tensor(flattened_flux, dtype=torch.float32)
    #     self.scale_factor = scale_factor

    def __init__(self, parquet_file, scale_factor=1000.0, cadence_minutes=29.4):
        self.df = pd.read_parquet(parquet_file)
        raw_flux = self.df['flux'].values
        
        # DYNAMIC DSP: Automatically calculate steps for a 25-hour moving window
        # Kepler (29.4 min) -> ~51 steps. TESS (2.0 min) -> 750 steps.
        window_steps = int((25 * 60) / cadence_minutes)
        if window_steps % 2 == 0: 
            window_steps += 1 # Rolling windows must be an odd number
            
        wave_baseline = self.df['flux'].rolling(window=window_steps, center=True, min_periods=1).median().values
        flattened_flux = (raw_flux - wave_baseline) * scale_factor
        
        self.flux = torch.tensor(flattened_flux, dtype=torch.float32)
        self.scale_factor = scale_factor

        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.flux.unsqueeze(-1)

if __name__ == "__main__":
    dataset = KeplerLightCurveDataset('data/Kepler-10_Q4.parquet')
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        print(f"Flattened & Scaled Tensor Shape: {batch.shape}")
        break
