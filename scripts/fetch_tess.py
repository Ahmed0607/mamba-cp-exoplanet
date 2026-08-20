import lightkurve as lk
import pandas as pd
import os

def main():
    print("Querying NASA's MAST archive for TESS observations of Pi Mensae...")
    
    # 1. Search for TESS short-cadence (2-minute) data processed by the standard SPOC pipeline
    search_result = lk.search_lightcurve("Pi Mensae", mission="TESS", author="SPOC")
    
    # 2. Download the very first observation sector available
    # NEW CODE
    print(f"Found {len(search_result)} observation sectors. Downloading the first available sector...")
    lc = search_result[0].download()
    
    # 3. Clean the raw telemetry (remove corrupted NaNs and extreme cosmic ray strikes)
    print("Applying initial DSP cleaning...")
    lc = lc.remove_nans().remove_outliers(sigma=5)
    
    # 4. Normalize the flux around 1.0 to match our existing dataset architecture
    median_flux = lc.flux.value.mean()
    normalized_flux = lc.flux.value / median_flux
    
    # 5. Package into a Pandas DataFrame
    df = pd.DataFrame({
        'time': lc.time.value,
        'flux': normalized_flux
    })
    
    # 6. Save as Parquet for the Dataloader
    os.makedirs('data', exist_ok=True)
    out_path = 'data/TESS_Pi_Mensae.parquet'
    df.to_parquet(out_path)
    
    print("-" * 40)
    print(f"Success! TESS sequence saved to {out_path}")
    print(f"Total continuous time steps: {len(df)}")
    print("-" * 40)

if __name__ == "__main__":
    main()
