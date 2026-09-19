# CS5787 HW1 - LeNet-5 on FashionMNIST

## Overview

This project implements a modified LeNet-5 model on the FashionMNIST dataset using PyTorch.

Four configurations are compared:

1. No Regularization
2. Dropout
3. Weight Decay
4. Batch Normalization

## Dataset

FashionMNIST contains 60,000 training images and 10,000 test images.

The original training set was split into:

- 54,000 training samples
- 6,000 validation samples

A random seed of 42 was used for reproducibility.

## Model Architecture

The LeNet-5 architecture contains:

- Conv2D: 1 → 6 channels
- Average Pooling
- Conv2D: 6 → 16 channels
- Average Pooling
- Fully Connected: 400 → 120
- Fully Connected: 120 → 84
- Output Layer: 84 → 10

ReLU activation is used throughout the network.

## Training Settings

- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 10
- Dropout rate: 0.5
- Weight decay: 1e-4

The model with the highest validation accuracy was saved for final evaluation.

The test set was not used for model selection.

## Training

Run the notebook from beginning to end:

`HW1.ipynb`

The notebook trains all four configurations and saves the best model weights.

## Saved Models

The following checkpoints are generated:

- `lenet5_base.pth`
- `lenet5_dropout.pth`
- `lenet5_weight_decay.pth`
- `lenet5_batchnorm.pth`

## Testing

To test a saved model, load the corresponding architecture and weights.

Example:

```python
model = LeNet5Base().to(device)

model.load_state_dict(
    torch.load(
        "lenet5_base.pth",
        map_location=device
    )
)

test_accuracy = compute_accuracy(
    model,
    test_loader
)

print(f"Test Accuracy: {test_accuracy:.2f}%")
## Final Results

| Technique | Train Accuracy | Test Accuracy |
|---|---:|---:|
| No Regularization | 92.32% | 89.70% |
| Dropout | 90.49% | 89.19% |
| Weight Decay | 91.16% | 88.99% |
| Batch Normalization | 95.57% | 90.91% |

All four configurations achieved more than 88% test accuracy.
