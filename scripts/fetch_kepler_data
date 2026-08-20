import lightkurve as lk
import os

def test_mast_connection():
    print("Searching MAST Archive for Kepler-10...")
    search_result = lk.search_lightcurve('Kepler-10', author='Kepler', cadence='long')
    print(f"Found {len(search_result)} continuous viewing quarters.")
    
    if len(search_result) > 0:
        print("Pipeline connection successful. Ready for mass ingestion.")

if __name__ == "__main__":
    test_mast_connection()
