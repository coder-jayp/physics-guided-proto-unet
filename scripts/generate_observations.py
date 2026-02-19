import os
import glob
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

def create_noisy_dataset(base_path):
    """
    Simulates observational effects (beam blur and thermal noise) 
    on standardized clean FITS files.
    """
    clean_dir = os.path.join(base_path, 'clean')
    noisy_dir = os.path.join(base_path, 'noisy')
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Search for our standardized clean files
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*_clean.fits")))
    
    if not clean_files:
        print(f"No clean files found in: {clean_dir}")
        print("Please run organize_data.py first.")
        return

    print(f"Injecting noise into {len(clean_files)} files...")

    for f_path in clean_files:
        with fits.open(f_path) as hdul:
            # Slicing the 5D cube: (Velo, Stokes, Freq, RA, Dec) -> (RA, Dec)
            # Standard input for our diffusion model preprocessing
            clean_data = hdul[0].data[0, 0, 0, :, :].astype(np.float32)
            header = hdul[0].header

        # --- STEP 1: RESOLUTION (Beam Blur) ---
        # sigma=2.0 simulates the instrument's Point Spread Function (PSF)
        beam_convolved = gaussian_filter(clean_data, sigma=2.0)

        # --- STEP 2: GAUSSIAN NOISE (Thermal Background) ---
        # Using a noise floor relative to typical flux values (~1.1e-21)
        noise_level = 1.1e-21 
        gaussian_noise = np.random.normal(0, noise_level, clean_data.shape)
        
        noisy_observation = beam_convolved + gaussian_noise

        # Save with matching sample index but 'noisy' suffix
        new_name = os.path.basename(f_path).replace("_clean.fits", "_noisy.fits")
        out_path = os.path.join(noisy_dir, new_name)
        
        # Update FITS header to reflect processed state
        header['DATATYPE'] = 'NOISY_OBSERVATION'
        header['SIGMA'] = (2.0, 'Beam blur sigma')
        
        fits.writeto(out_path, noisy_observation, header, overwrite=True)
        print(f"Created: {new_name}")

    print(f"\nSuccess! Noisy dataset populated at: {noisy_dir}")

if __name__ == "__main__":
    # Robust pathing: find project root relative to this script's location
    current_script_path = os.path.abspath(__file__)
    scripts_folder = os.path.dirname(current_script_path)
    project_root = os.path.dirname(scripts_folder)
    
    # Point to the data directory
    DATA_PATH = os.path.join(project_root, "data")
    
    if os.path.exists(DATA_PATH):
        create_noisy_dataset(DATA_PATH)
    else:
        print(f"Error: Data path not found at {DATA_PATH}")