import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchio as tio
import glob

# Paths
brats_root = r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\dataset\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
ixi_root = r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\dataset\ixi"
output_dir = r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\combined_dataset"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

# Preprocessing function
def preprocess_nifti(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    data = np.clip(data, np.percentile(data, 1), np.percentile(data, 99))
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = torch.tensor(data).unsqueeze(0).float()  # [1, D, H, W]
    transform = tio.Resize((128, 128, 128))
    data = transform(data)
    return data.squeeze().numpy()

# Output list
records = []
idx = 1

# Process BraTS (tumor)
print("🔴 Processing BraTS subjects...")
for folder in tqdm(os.listdir(brats_root)):
    sub_path = os.path.join(brats_root, folder)
    if not os.path.isdir(sub_path):
        continue

    flair_candidates = glob.glob(os.path.join(sub_path, "*flair.nii*"))
    seg_candidates = glob.glob(os.path.join(sub_path, "*seg.nii*"))

    if flair_candidates and seg_candidates:
        flair_file = flair_candidates[0]
        seg_file = seg_candidates[0]

        try:
            flair = preprocess_nifti(flair_file)
            seg = nib.load(seg_file).get_fdata()
            tumor_voxels = np.any(seg > 0)

            out_path = os.path.join(output_dir, "images", f"subject_{idx:03d}.nii.gz")
            nib.save(nib.Nifti1Image(flair, affine=np.eye(4)), out_path)

            label = 1 if tumor_voxels else 0
            records.append({"filename": f"subject_{idx:03d}.nii.gz", "label": label})
            idx += 1
        except Exception as e:
            print(f"❌ Error processing {folder}: {e}")
    else:
        print(f"⚠️ Skipping {folder} - Missing flair or seg file.")

# Process IXI (healthy)
print("🟢 Processing IXI healthy subjects...")
for file in tqdm(os.listdir(ixi_root)):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        try:
            path = os.path.join(ixi_root, file)
            data = preprocess_nifti(path)
            out_path = os.path.join(output_dir, "images", f"subject_{idx:03d}.nii.gz")
            nib.save(nib.Nifti1Image(data, affine=np.eye(4)), out_path)
            records.append({"filename": f"subject_{idx:03d}.nii.gz", "label": 0})
            idx += 1
        except Exception as e:
            print(f"❌ Error processing IXI file {file}: {e}")

# Save labels.csv
df = pd.DataFrame(records)
df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

print("✅ Dataset preparation complete!")
