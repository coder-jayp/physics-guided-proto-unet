import torch
import numpy as np
import ctypes
import os
import matplotlib.pyplot as plt
from models.unet import UNet
from scripts.data_loader import get_dataloaders

project_root = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.add_dll_directory(r"C:\w64devkit\bin")

dll_path = os.path.join(project_root, 'physics_engine', 'profile.dll')
model_path = os.path.join(project_root, "checkpoints", "best_physics_model.pth")
CSV_PATH = os.path.join(project_root, "data", "dataset.csv")
DATA_ROOT = os.path.join(project_root, "data")

if not os.path.exists(dll_path):
    raise FileNotFoundError(f"Could not find DLL at {dll_path}")

fortran_lib = ctypes.CDLL(dll_path)
fortran_lib.calculate_radial_profile.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), 
    ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, 
    ctypes.POINTER(ctypes.c_int)
]

def run_fortran_physics(img_array):
    nx_val, ny_val = img_array.shape
    n_bins_val = 100
    img_input = np.asfortranarray(img_array, dtype=np.float32)
    profile = np.zeros(n_bins_val, dtype=np.float32)
    nx, ny, n_bins = ctypes.c_int(nx_val), ctypes.c_int(ny_val), ctypes.c_int(n_bins_val)
    fortran_lib.calculate_radial_profile(img_input.ctypes.data, ctypes.byref(nx), 
                                         ctypes.byref(ny), profile.ctypes.data, ctypes.byref(n_bins))
    return profile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(DEVICE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing weight file: {model_path}. Please run train.py first.")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

_, val_loader = get_dataloaders(CSV_PATH, DATA_ROOT, batch_size=1)
noisy_img, clean_img = next(iter(val_loader))

with torch.no_grad():
    prediction = model(noisy_img.to(DEVICE))
    denoised_np = prediction.cpu().squeeze().numpy()

noisy_np = noisy_img.squeeze().numpy()
clean_np = clean_img.squeeze().numpy()

print("📊 Invoking Fortran Engine for Radial Profiling...")
profile_noisy = run_fortran_physics(noisy_np)
profile_denoised = run_fortran_physics(denoised_np)
profile_clean = run_fortran_physics(clean_np)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(noisy_np, cmap='magma'); ax[0].set_title("Input (Noisy Observation)")
ax[1].imshow(denoised_np, cmap='magma'); ax[1].set_title("Output (U-Net Denoised)")
ax[2].imshow(clean_np, cmap='magma'); ax[2].set_title("Target (Ground Truth)")
for a in ax: a.axis('off')



plt.figure(figsize=(12, 6))
plt.plot(profile_noisy, label="Noisy Profile (Pre-Correction)", color='gray', alpha=0.4)
plt.plot(profile_denoised, label="AI Reconstructed Profile", color='cyan', linewidth=2.5)
plt.plot(profile_clean, label="Ground Truth (Physics Model)", color='orange', linestyle='--')
plt.xlabel("Radial Distance (Bins)")
plt.ylabel("Normalized Intensity")
plt.title("Physical Consistency Check: Radial Brightness Distribution")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

print("Analysis complete. Displaying results...")
plt.show()