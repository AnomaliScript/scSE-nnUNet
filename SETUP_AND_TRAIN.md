# scSE nnUNet Setup and Training Guide

This guide will help you set up and test the scSE-enhanced nnUNet with bottleneck and decoder attention.

## Environment Setup

The preprocessed data is located in:
```
C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\
├── nnUNet_raw/           # Raw dataset (51 cases)
├── nnUNet_preprocessed/  # Preprocessed data (ready to use)
└── nnUNet_results/       # Training outputs will go here
```

## Option 1: Manual Environment Variable Setup (Windows CMD)

Open a new Command Prompt and run:

```batch
set nnUNet_raw=C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_raw
set nnUNet_preprocessed=C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_preprocessed
set nnUNet_results=C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_results
```

## Option 2: Use the Batch Script (Recommended for Windows)

Double-click or run:
```
setup_env_and_train.bat
```

This will set the environment variables and keep a command prompt open with them configured.

## Option 3: Use Python Script to Set Variables

Run in conda environment:
```bash
cd C:\Users\anoma\Downloads\scse-nnUNet
conda activate base
python set_nnunet_env.py
```

---

## Testing the Installation

Before training, verify everything works:

```bash
cd C:\Users\anoma\Downloads\scse-nnUNet
conda activate base
python test_scse_network.py
```

This test script will:
1. Import the custom trainer
2. Load the dataset configuration
3. Initialize the network with bottleneck + decoder scSE attention
4. Test a forward pass with dummy data

If all tests pass, you'll see:
```
✓ All tests passed! scSE nnUNet is ready for training.
```

---

## Training Commands

### Full Training (Fold 0, with validation)

After setting environment variables, run:

```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc
```

Parameters:
- `001`: Dataset ID (Dataset001_Cervical)
- `3d_fullres`: Configuration (3D full resolution)
- `0`: Fold number (0-4 for 5-fold cross-validation)
- `-tr nnUNetTrainerCervicalAttentionResEnc`: Custom trainer with scSE attention

### Training with Specific GPU

```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc --device cuda:0
```

### Continue Training from Checkpoint

```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc --c
```

### Training All Folds (Cross-Validation)

Train each fold separately:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc
nnUNetv2_train 001 3d_fullres 1 -tr nnUNetTrainerCervicalAttentionResEnc
nnUNetv2_train 001 3d_fullres 2 -tr nnUNetTrainerCervicalAttentionResEnc
nnUNetv2_train 001 3d_fullres 3 -tr nnUNetTrainerCervicalAttentionResEnc
nnUNetv2_train 001 3d_fullres 4 -tr nnUNetTrainerCervicalAttentionResEnc
```

---

## What You'll See During Training

When training starts, you'll see these messages confirming the architecture:

```
🎯 Building ResidualEncoderUNet with Bottleneck + Decoder Cervical Attention...

📌 BOTTLENECK ATTENTION:
  ✅ Bottleneck: 320 channels

📌 DECODER BLOCK ATTENTION:
  ✅ Decoder 0: 256 channels
  ✅ Decoder 1: 128 channels
  ✅ Decoder 2: 64 channels
  ✅ Decoder 3: 32 channels
  ✅ Decoder 4: 16 channels

✅ Bottleneck + Decoder Cervical Attention integration complete!
   Bottleneck attention: ✓
   Decoder attention modules: 5
   Skip connections: UNCHANGED (direct gradient flow)
   Architecture: ResidualEncoderUNet

🎯 Class-weighted loss configured:
   C1-C5: 1.0x weight (standard)
   C6-C7: 2.0x weight (prioritized)
```

---

## Architecture Features

Your model includes:

1. **Bottleneck scSE Attention**
   - Applied to the deepest encoder features
   - Refines global context understanding

2. **Decoder Block scSE Attention**
   - Applied after each decoder convolution block
   - Improves boundary refinement during upsampling

3. **Anatomical Routing (YOLO-guided)**
   - C1: Spatial attention only (ring structure)
   - C2: Enhanced scSE (odontoid process)
   - C3-C7: Standard scSE

4. **Class-Weighted Loss**
   - 2x penalty for C6/C7 errors
   - Focuses training on challenging vertebrae

5. **Skip Connections Unchanged**
   - Direct gradient flow preserved
   - Standard U-Net benefits maintained

---

## Training Output Location

Results will be saved to:
```
C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_results\
└── Dataset001_Cervical/
    └── nnUNetTrainerCervicalAttentionResEnc__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            ├── progress.png
            ├── training_log.txt
            └── validation_raw/
```

---

## Monitoring Training

Training progress is logged to:
- Console output (real-time)
- `training_log.txt` (detailed log)
- `progress.png` (loss/metric plots)

Key metrics tracked:
- Dice score per class (C1-C7)
- Overall mean Dice
- Training and validation loss

---

## Dataset Information

- **Total cases**: 51 (42 train, 9 validation in fold 0)
- **Labels**: 8 classes (background + C1-C7)
- **Modality**: CT scans
- **Preprocessing**: nnUNet z-score normalization + resampling

---

## Troubleshooting

### "Environment variable not set" error
Make sure you've run the setup batch file or manually set the variables in your current terminal session.

### "Dataset not found" error
Verify the paths:
```bash
dir C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_preprocessed\Dataset001_Cervical
```

### CUDA out of memory
Reduce batch size by modifying the plans file or use a smaller patch size configuration.

### YOLO model not found
Verify the YOLO model is at:
```
C:\Users\anoma\Downloads\scse-nnUNet\nnunetv2\yolo_models\vertebra_detector_114.pt
```

---

## Next Steps After Training

1. **Evaluate on validation set**:
   ```bash
   nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -tr nnUNetTrainerCervicalAttentionResEnc -f 0
   ```

2. **Compare with baseline nnUNet**:
   Train a standard nnUNet for comparison
   ```bash
   nnUNetv2_train 001 3d_fullres 0
   ```

3. **Analyze class-specific performance**:
   Check the Dice scores for C6 and C7 specifically to see if the weighted loss helped

---

## Questions?

- Architecture details: See `nnUNetTrainerCervicalAttentionResEnc.py`
- scSE modules: See `scse_modules.py`
- YOLO attention generation: See `generate_yolo_attention_mask()` in `scse_modules.py`
