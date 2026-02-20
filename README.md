# Physics-Guided Proto-UNet 🪐

This project integrates a PyTorch U-Net with a high-performance Fortran engine to enforce radial profile consistency during training.

---

## 🚀 Overview
Astronomical observations of protoplanetary disks are often limited by thermal noise and beam convolution. While standard Convolutional Neural Networks (CNNs) can denoise images, they often lose physical structural integrity, such as the exact locations of gaps and ring widths.

This project implements a **Physics-Guided Loss**:
1. **Pixel-wise Loss (MSE):** Ensures the denoised image looks like the ground truth.
2. **Radial Profile Loss:** A Fortran-backed engine calculates the azimuthal average (radial profile) to ensure the denoised output respects the physical brightness distribution of the disk model.



---

## 🛠️ Architecture
The pipeline consists of a hybrid software stack:
- **Deep Learning:** PyTorch (U-Net with DoubleConv blocks and Bilinear interpolation).
- **Physics Engine:** Fortran 90 (Parallelized with **OpenMP** for high-speed radial profiling).
- **Interface:** `ctypes` bridge for direct memory mapping between NumPy arrays and Fortran pointers.



---

## 📂 Project Structure
* `train.py`: Main training loop featuring the hybrid Physics/Pixel loss.
* `final_results.py`: Inference script for visualizing denoised disks and comparing radial profiles.
* `models/unet.py`: Standard U-Net architecture optimized for $600 \times 600$ astronomical data.
* `physics_engine/profile.f90`: High-performance radial profile calculator source code.
* `scripts/`: 
    * `generate_observations.py`: Simulates telescope effects (Beam blur + Thermal noise).
    * `organize_data.py`: Standardizes dataset manifests.
    * `data_loader.py`: Handles FITS file loading and log-scaling.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- PyTorch (CUDA recommended)
- GFortran (via **w64devkit** for Windows users)

### 2. Compile Physics Engine
Navigate to the `physics_engine` folder and compile the shared library:
```powershell
gfortran -O3 -fopenmp -shared -o profile.dll profile.f90

### 3. Prepare Data
Ensure your clean FITS files are in `data/clean`, then run:

```powershell
python scripts/generate_observations.py
python scripts/organize_data.py

---

## 📊 Results & Validation
The model is validated not just on pixel error, but on **Physical Consistency**. By plotting the radial brightness distribution, we can verify if the AI-reconstructed disk maintains the correct astrophysical structure.