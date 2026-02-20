import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ctypes
import os
from models.unet import UNet
from scripts.data_loader import get_dataloaders

project_root = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':
    try:
        os.add_dll_directory(r"C:\w64devkit\bin")
    except Exception as e:
        print(f"Note: Could not add w64devkit to path: {e}")

dll_path = os.path.join(project_root, 'physics_engine', 'profile.dll')
CSV_PATH = os.path.join(project_root, "data", "dataset.csv")
DATA_ROOT = os.path.join(project_root, "data")

fortran_lib = None
if os.path.exists(dll_path):
    try:
        fortran_lib = ctypes.CDLL(dll_path)
        fortran_lib.calculate_radial_profile.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), 
            ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_int)
        ]
        print("✅ Physics Engine (Fortran) loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Found DLL but failed to load: {e}")

def get_physics_profile(img_tensor):
    if fortran_lib is None: return None
    img_np = img_tensor.detach().cpu().numpy().squeeze()
    nx_val, ny_val = img_np.shape
    n_bins_val = 100
    img_fortran = np.asfortranarray(img_np, dtype=np.float32)
    profile = np.zeros(n_bins_val, dtype=np.float32)
    nx, ny, n_bins = ctypes.c_int(nx_val), ctypes.c_int(ny_val), ctypes.c_int(n_bins_val)
    fortran_lib.calculate_radial_profile(
        img_fortran.ctypes.data, ctypes.byref(nx), 
        ctypes.byref(ny), profile.ctypes.data, ctypes.byref(n_bins)
    )
    return torch.from_numpy(profile).to(img_tensor.device)

# --- 2. Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 
EPOCHS = 30
LEARNING_RATE = 1e-4
PHYSICS_LAMBDA = 0.05 

CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader, val_loader = get_dataloaders(CSV_PATH, DATA_ROOT, batch_size=BATCH_SIZE)

model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_loss = float('inf')

print(f"🚀 Starting Physics-Guided Training on {device}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_total_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss_pixel = criterion(outputs, clean)
        loss_physics = 0
        if fortran_lib:
            prof_pred = get_physics_profile(outputs[0])
            prof_true = get_physics_profile(clean[0])
            loss_physics = criterion(prof_pred, prof_true)
        
        total_loss = loss_pixel + (PHYSICS_LAMBDA * loss_physics)
        total_loss.backward()
        optimizer.step()
        train_total_loss += total_loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            preds = model(noisy)
            val_loss += criterion(preds, clean).item()
    
    avg_train = train_total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_physics_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"⭐ New best model saved (Val Loss: {avg_val:.6f})")

print("\Training Complete!")