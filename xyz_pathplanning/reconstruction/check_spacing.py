
import nibabel as nib
import numpy as np

volume_path = "C:/Users/anoma/Downloads/spine-segmentation-data-cleaning/CTSpine1K/clean_volumes/CTS1K_007_0000.nii.gz"
print(f"Checking: {volume_path}")

img = nib.load(volume_path)
header = img.header
zooms = header.get_zooms()

print(f"\nVoxel Spacing (Zooms): {zooms}")
print(f"  X spacing: {zooms[0]:.4f} mm")
print(f"  Y spacing: {zooms[1]:.4f} mm")
print(f"  Z spacing: {zooms[2]:.4f} mm (Slice Thickness)")

print(f"\nData Shape: {img.shape}")
