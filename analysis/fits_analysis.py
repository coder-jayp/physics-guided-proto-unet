import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (astropy_mpl_style, ImageNormalize, 
                                   LogStretch, PercentileInterval)

plt.style.use(astropy_mpl_style)

def analyze_disk_physics(file_path):
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        data = np.squeeze(hdul[0].data)
        if data.ndim > 2:
            data = data[0]

        bmaj = header.get('BMAJ', header.get('CDELT1', 0.0))
        bmin = header.get('BMIN', header.get('CDELT2', 0.0))
        unit = header.get('BUNIT', 'W.m-2.pixel-1')
        
        d_min, d_max = np.nanmin(data), np.nanmax(data)
        dyn_range = np.log10(d_max / (abs(d_min) + 1e-25))

        print(f"Dimensions:    {data.shape}")
        print(f"Physical Beam: {bmaj:.2e} x {bmin:.2e}")
        print(f"Dynamic Range: 10^{dyn_range:.1f} orders of magnitude")
        print(f"Max Flux:      {d_max:.2e} {unit}")

        # --- Linear Intensity ---
        plt.figure("Linear Scale Analysis", figsize=(8, 8))
        plt.imshow(data, cmap='magma', origin='lower')
        plt.title("Linear Intensity (Raw Observation View)", pad=15)
        plt.axis('off')
        plt.tight_layout()

        # --- Log Scale Substructures ---
        plt.figure("Log-Physical Substructures", figsize=(9, 8))
        norm = ImageNormalize(data, interval=PercentileInterval(99.7), stretch=LogStretch())
        im = plt.imshow(data, cmap='magma', origin='lower', norm=norm)
        plt.title("Log-Stretched Physics (Revealing Gaps & Rings)", pad=15)
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(f'Normalized Intensity ({unit})', rotation=270, labelpad=20)
        plt.tight_layout()

        # --- Voxel Distribution ---
        plt.figure("Intensity Distribution Diagnostic", figsize=(8, 6))
        filtered = data.flatten()[data.flatten() > 0]
        plt.hist(filtered, bins=np.logspace(np.log10(filtered.min()), np.log10(filtered.max()), 60), 
                 color='crimson', edgecolor='black', alpha=0.8)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Voxel Intensity Distribution (Log-Log)", pad=15)
        plt.xlabel(f"Flux ({unit})")
        plt.ylabel("Pixel Count")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        # Show all frames simultaneously
        print("\nDisplaying frames... Close windows to terminate.")
        plt.show()

if __name__ == "__main__":
    base_path = r"D:\physics-guided-proto-unet"
    fits_files = glob.glob(os.path.join(base_path, "*.fits"))

    if fits_files:
        analyze_disk_physics(fits_files[0])
    else:
        print("No FITS files found.")