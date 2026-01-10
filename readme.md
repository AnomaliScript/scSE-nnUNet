# scSE-nnUNet: Cervical Spine Segmentation with Attention Mechanisms
Modified version of nnUNet exploring optimal attention placement for cervical spine segmentation.

## Project Evolution

### Initial Hypothesis (Not Implemented)
Originally planned cervical level-aware attention with vertebra-specific pathways (C1/C2/C3-C7). 
Due to **the failure of implementing YOLOv11 to serve as a classifier for vertebrae because of limited high-quality cervical data to train both the YOLO classifier and the origina scSE-nnUNet**, this approach was not pursued.

### Actual Implementation
Systematic evaluation of scSE attention block placement within nnUNet architecture:

1. **Baseline**: ResEnc nnUNet (no attention) - 0.83 DSC
2. **Configuration 1**: scSE in skip connections only - 0.83 DSC (no improvement)
3. **Configuration 2**: scSE in decoder + bottleneck - 0.88 DSC (+5% improvement)

## Key Findings
- Skip connection attention alone did not improve segmentation performance
- Decoder + bottleneck attention significantly improved C6/C7 boundary precision
- Lower cervical vertebrae (C6/C7) showed greatest improvement (+4.2% DSC)
- Demonstrates importance of attention placement for boundary refinement tasks

## Technical Details
- Based on nnU-Net v2 architecture
- Integrated scSE (concurrent Spatial and Channel Squeeze-and-Excitation) blocks
- Focus on cervical vertebrae (C1-C7) segmentation from CT scans

## Results
| Configuration | Overall DSC | C7 DSC |
|--------------|-------------|---------|
| Baseline (no attention) | 0.830 | 0.840 |
| Skip connections | 0.830 | 0.840 |
| Decoder + Bottleneck | 0.880 | 0.882 |

## Original Work
This project is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) by Fabian Isensee et al.

## License
This project maintains the original Apache 2.0 License from nnUNet. See LICENSE file.

Modifications and additions © 2025 Brandon Kim

## Citations
If you use this work, please cite:
- Isensee et al. - nnU-Net (original architecture)
- Roy et al. (2018) - scSE attention mechanisms
