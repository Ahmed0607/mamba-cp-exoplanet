import lightkurve as lk
import pandas as pd
import os

def fetch_and_clean_target(target_name, quarter=4):
    print(f"Downloading {target_name} Quarter {quarter}...")
    
    # 1. Download the LightCurve file
    search_result = lk.search_lightcurve(target_name, author='Kepler', cadence='long', quarter=quarter)
    lc = search_result.download()
    
    # 2. Clean the data (Remove missing observation frames)
    lc_clean = lc.remove_nans()
    
    # 3. Normalize the flux (centers the baseline stellar brightness at 1.0)
    lc_norm = lc_clean.normalize()
    
    # 4. Extract time and flux as 1D arrays
    time = lc_norm.time.value
    flux = lc_norm.flux.value
    
    print(f"Original data points: {len(lc.flux)}")
    print(f"Cleaned data points: {len(flux)}")
    
    # 5. Save to the highly efficient Parquet format for GPU loading
    df = pd.DataFrame({'time': time, 'flux': flux})
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    save_path = f"data/{target_name}_Q{quarter}.parquet"
    df.to_parquet(save_path, engine='pyarrow')
    
    print(f"Saved optimized time-series tensor to: {save_path}")

if __name__ == "__main__":
    fetch_and_clean_target('Kepler-10', quarter=4)
