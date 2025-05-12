# SpaceNet 9 Challenge

## Cross-Modal Image Registration with Deep Learning

The SpaceNet 9 challenge focuses on developing algorithms for accurate pixel-wise cross-modal satellite image registration. The task is to precisely align Synthetic Aperture Radar (SAR) imagery with optical satellite imagery, producing a continuous displacement map that specifies how each pixel in the SAR image should be transformed to match the optical image.

For detailed information about the challenge, see [docs/challenge.md](docs/challenge.md) and [docs/paper.md](docs/paper.md).

## Dataset

The dataset consists of paired optical (RGB) and SAR (single-channel) satellite images:
- Training data: `train.zip`
- Test data: `publictest.zip`
- Baseline submission: `baseline-submission.zip`
- Trivial submission: `trivial-submission.zip`
- Perfect solution: `perfect-solution.zip`
- Scorer: `scorer.zip`

Download links are configured in [configs/data_urls.yaml](configs/data_urls.yaml).

## Task Description

1. For each pair of optical and SAR images, compute the continuous pixel-wise spatial transformations (x-shift and y-shift) required to align the SAR image with the optical image.
2. Output is a two-channel transformation image at the same resolution as the optical image, with channels representing:
   - Channel 1: X-shift (horizontal displacement)
   - Channel 2: Y-shift (vertical displacement)
3. The algorithm is evaluated based on the accuracy of tie-point transformations.

## Approach

Our solution will be based on deep learning for dense correspondence estimation between optical and SAR imagery. We'll explore:

1. **Dataset Implementation**: Using TorchGeo and TerraTorch conventions to build a custom dataset for SpaceNet 9
2. **Model Architecture**: Leveraging TerraTorch's foundation models and adapting them for cross-modal registration
3. **Training Pipeline**: Implementing a PyTorch Lightning training pipeline for the registration task
4. **Evaluation Metrics**: Using the provided scorer to evaluate the registration performance

## Repository Structure

```
spacenet-9/
├── data/
│   ├── raw/          # Raw SpaceNet 9 data
│   └── processed/    # Preprocessed data for training
├── src/
│   ├── datasets/     # Dataset implementations
│   ├── models/       # Model architectures
│   ├── tasks/        # PyTorch Lightning tasks
│   └── utils/        # Utility functions
├── configs/          # Configuration files
├── scripts/          # Training and inference scripts
├── docs/             # Challenge documentation and research papers
│   └── images/       # Documentation images
├── torchgeo/         # TorchGeo reference implementation (submodule)
└── terratorch/       # TerraTorch reference implementation (submodule)
```