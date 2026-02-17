import os
import shutil
import glob
import pandas as pd
from astropy.io import fits

def organize_with_metadata(project_root):
    raw_dir = os.path.join(project_root, 'data', 'raw')
    clean_dir = os.path.join(project_root, 'data', 'clean')
    
    os.makedirs(clean_dir, exist_ok=True)
    
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.fits")))
    metadata_list = []

    if not raw_files:
        print(f"No files found in: {raw_dir}")
        print("Please ensure your .fits files are placed in 'data/raw/'")
        return

    print(f"Standardizing {len(raw_files)} files...")

    for i, old_path in enumerate(raw_files):
        new_filename = f"sample_{i:03d}_clean.fits"
        new_path = os.path.join(clean_dir, new_filename)

        shutil.copy(old_path, new_path)
        
        try:
            with fits.open(new_path, mode='update') as hdul:
                hdul[0].header['DATA_TYP'] = ('CLEAN_SAMPLE', 'Standardized scientific data')
                if 'ORIG_ID' in hdul[0].header:
                    del hdul[0].header['ORIG_ID']
                
                hdul.flush()
        except Exception as e:
            print(f"Warning: Could not process file index {i}: {e}")

        metadata_list.append({
            "sample_name": new_filename
        })
        
        print(f"Standardized: {new_filename}")

    df = pd.DataFrame(metadata_list)
    inventory_path = os.path.join(project_root, "data", "dataset.csv")
    df.to_csv(inventory_path, index=False)
    
    print(f"\nSuccess! Organized data and inventory created at: {inventory_path}")

if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    scripts_folder = os.path.dirname(current_script_path)
    project_root = os.path.dirname(scripts_folder)
    
    organize_with_metadata(project_root)