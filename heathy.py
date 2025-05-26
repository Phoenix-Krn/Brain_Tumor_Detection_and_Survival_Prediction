import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm

# Set your healthy dataset folder (adjust this)
# Set the root directory where all subfolders like BRAINIX, KIRBY21, etc., are stored
healthy_data_root = r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\Neurohacking_data-0.0"

# Output folder where preprocessed files will go
output_dir = r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\healthy_preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Search for FLAIR files recursively
flair_paths = list(Path(healthy_data_root).rglob("*FLAIR.nii.gz"))

def preprocess_image(path, target_shape=(128, 128, 128)):
    img = nib.load(str(path))
    data = img.get_fdata()

    # Normalize and convert to tensor
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    data = torch.tensor(data).float()

    # Resize to target shape with padding or cropping
    data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
    data = F.interpolate(data, size=target_shape, mode='trilinear', align_corners=False)
    return data.squeeze(0).numpy()

def save_nifti(data, reference_path, output_path):
    ref_img = nib.load(str(reference_path))
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, output_path)

print(f"Found {len(flair_paths)} healthy FLAIR files.")

for i, flair_path in enumerate(tqdm(flair_paths)):
    subject_id = f"healthy_{i:03d}"
    
    # Process image
    processed = preprocess_image(flair_path)

    # Save image in BraTS-style format
    out_img_path = os.path.join(output_dir, f"{subject_id}_0000.nii.gz")
    out_seg_path = os.path.join(output_dir, f"{subject_id}_seg.nii.gz")

    save_nifti(processed, flair_path, out_img_path)

    # Save dummy segmentation mask (all zeros)
    dummy_mask = np.zeros(processed.shape, dtype=np.uint8)
    save_nifti(dummy_mask, flair_path, out_seg_path)

print("✅ Preprocessing complete. Healthy images ready for training!")
