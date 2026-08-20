import pandas as pd
import matplotlib.pyplot as plt

def plot_lightcurve(file_path):
    print(f"Loading tensor from {file_path}...")
    
    # Load the Parquet file back into a DataFrame
    df = pd.read_parquet(file_path)
    
    # Set up the plot with a clean, dark-mode aesthetic
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 5))
    
    # Plot the time vs. flux
    plt.scatter(df['time'], df['flux'], s=2, color='cyan', alpha=0.7)
    
    plt.title('Kepler-10 Quarter 4: Normalized Stellar Flux')
    plt.xlabel('Time (Barycentric Julian Date)')
    plt.ylabel('Normalized Flux')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    print("Generating plot...")
    plt.show()

if __name__ == "__main__":
    plot_lightcurve('data/Kepler-10_Q4.parquet')
