import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class ProtoUnetDataset(Dataset):
    """
    Handles FITS loading, log-scaling, and normalization.
    """
    def __init__(self, dataframe, root_dir):
        self.metadata = dataframe
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata)

    def _load_fits(self, path, is_clean=True):
        with fits.open(path) as hdul:
            # Original clean data is 5D; noisy was saved as 2D
            data = hdul[0].data
            if is_clean:
                # Slicing the 5D cube: (Velo, Stokes, Freq, RA, Dec) -> (RA, Dec)
                data = data[0, 0, 0, :, :]
            
        # 1. Physical Scaling (Standardize based on expected 10^-20 flux range)
        data = data * 1e20 
        
        # 2. Log Scaling to handle high dynamic range in disk structures
        # We use log1p to stay robust against small negative values from noise
        data = np.log1p(np.abs(data))
        
        return torch.from_numpy(data).float().unsqueeze(0) # Returns [1, H, W]

    def __getitem__(self, idx):
        # Updated to use the privacy-safe column 'sample_name'
        clean_name = self.metadata.iloc[idx]['sample_name']
        noisy_name = clean_name.replace('_clean.fits', '_noisy.fits')

        clean_path = os.path.join(self.root_dir, 'clean', clean_name)
        noisy_path = os.path.join(self.root_dir, 'noisy', noisy_name)

        clean_img = self._load_fits(clean_path, is_clean=True)
        noisy_img = self._load_fits(noisy_path, is_clean=False)

        # Final Normalization to [-1, 1] range for Diffusion Model stability
        clean_img = (clean_img / (clean_img.max() + 1e-8)) * 2 - 1
        noisy_img = (noisy_img / (noisy_img.max() + 1e-8)) * 2 - 1

        return noisy_img, clean_img

def get_dataloaders(csv_path, root_dir, batch_size=4, train_size=0.8):
    """
    Prepares train and validation DataLoaders.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest not found at {csv_path}. Run organize_data.py first.")

    df = pd.read_csv(csv_path)
    
    # Switched to train_test_split since specific 'system_id' clues are now removed
    train_df, val_df = train_test_split(df, train_size=train_size, random_state=42, shuffle=True)
    
    # Create Dataset objects
    train_ds = ProtoUnetDataset(train_df, root_dir)
    val_ds = ProtoUnetDataset(val_df, root_dir)
    
    # Create DataLoader objects
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Data Loading Summary:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Val samples:   {len(val_df)}")
    
    return train_loader, val_loader

# Testing block
if __name__ == "__main__":
    # Get project root (moves up from scripts folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    CSV_PATH = os.path.join(project_root, "data", "dataset.csv")
    DATA_ROOT = os.path.join(project_root, "data")
    
    try:
        train_loader, val_loader = get_dataloaders(CSV_PATH, DATA_ROOT)
        
        # Take a peek at one batch
        noisy_batch, clean_batch = next(iter(train_loader))
        print(f"\nBatch Loaded Successfully!")
        print(f"Noisy Batch Shape: {noisy_batch.shape}") 
        print(f"Clean Batch Shape: {clean_batch.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")