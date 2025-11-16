# scSE-nnUNet: Cervical Spine Segmentation with Level-Aware Attention

Modified version of nnUNet with cervical-specific attention mechanisms for surgical path planning.

## Novel Contributions
- Cervical level-aware attention mechanism (C1, C2, C3-C7)
- Integration of scSE blocks in skip connections
- Specialized architecture for vertebrae segmentation

## Original Work
This project is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) by Fabian Isensee et al.

## License
This project maintains the original Apache 2.0 License from nnUNet. See LICENSE file.

Modifications and additions © 2024 [Your Name]

## Citation
If you use this work, please cite both:
- The original nnUNet paper (Isensee et al.)
- Roy et al. (2018) for scSE attention mechanisms