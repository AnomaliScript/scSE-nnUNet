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

Modifications and additions © 2025 Brandon Kim

## Citation
If you use this work, please cite both:
- The original nnUNet paper (Isensee et al.)
- Roy et al. (2018) for scSE attention mechanisms



### v1.1: Cervical Level-Aware Attention with Soft Weighting

#### Context

Cervical vertebrae exhibit distinct anatomical characteristics:
- **C1 (atlas)**: Lacks a vertebral body; has a unique ring structure
- **C2 (axis)**: Features the odontoid process (dens), a distinctive superior projection
- **C3-C7**: Share similar rectangular vertebral body morphology

The past version treated all vertebrae uniformly, ignoring these anatomical 
differences. This version addresses this by applying vertebra-specific attention pathways.

#### Technical Implementation

The system consists of three components:

1. **Vertebra-Level Classifier**: A lightweight neural network that analyzes encoder 
   features and predicts probability distributions over three vertebra groups: 
   P(C1), P(C2), P(C3-C7)

2. **Anatomically-Informed Attention Pathways**:
   - **C1 pathway**: Spatial attention only (emphasizes structural relationships)
   - **C2 pathway**: Enhanced scSE with increased capacity (captures unique dens features)
   - **C3-C7 pathway**: Standard scSE (sufficient for similar anatomy)

3. **Soft Weighting Mechanism**: Rather than using hard selection (if-else logic), 
   the system blends all three pathway outputs using the predicted probabilities:
```
   Output = P(C1) × C1_attention + P(C2) × C2_attention + P(C3-C7) × C3-C7_attention
```

#### Advantages of Soft Weighting

- **Robustness**: Handles classifier uncertainty gracefully (e.g., if 60% confident 
  in C2, still applies 40% of other pathways)
- **Mixed Regions**: Naturally handles cases where multiple vertebrae appear in the 
  same receptive field
- **Training Stability**: Allows gradients to flow through all pathways, enabling 
  continuous learning across all attention mechanisms
- **No Hard Boundaries**: Avoids abrupt transitions between attention types

#### Integration with nnU-Net

The attention modules are integrated into nnU-Net's skip connections, following 
Roy et al.'s finding that skip connection integration outperforms post-encoder placement 
for medical imaging tasks. This allows the decoder to receive anatomically-refined 
features that emphasize relevant characteristics for each vertebra type.

#### Learning Process

During training, the vertebra-level classifier learns to identify cervical levels 
through the segmentation loss signal. When the classifier misidentifies a vertebra 
(e.g., predicts C1 but ground truth shows C2), the resulting poor segmentation 
produces high loss, and backpropagation adjusts the classifier weights to improve 
future predictions. The soft weighting ensures that even imperfect classifications 
still contribute appropriate attention, preventing catastrophic failures from 
misclassification.
