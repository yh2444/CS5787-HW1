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

Run `HW1.ipynb` from beginning to end to reproduce all experiments.

The notebook trains four configurations separately:

1. **No Regularization**
   - Uses the base LeNet-5 model.
   - No dropout, weight decay, or batch normalization.

2. **Dropout**
   - Uses `LeNet5Dropout`.
   - Dropout with probability `p = 0.5` is applied after the first two fully connected hidden layers.

3. **Weight Decay**
   - Uses the base LeNet-5 architecture.
   - L2 regularization is applied through Adam with `weight_decay = 1e-4`.

4. **Batch Normalization**
   - Uses `LeNet5BatchNorm`.
   - Batch normalization is applied after the convolutional and fully connected hidden layers.

Each configuration is trained for 10 epochs. After each epoch, validation accuracy is computed. The checkpoint with the highest validation accuracy is saved and later used for final train and test evaluation.

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

## Results

The following figure shows the training and test accuracy over 10 epochs for all four LeNet-5 configurations.

![LeNet-5 FashionMNIST Convergence](lenet5_convergence.png)

### Final Accuracy

| Technique | Best Epoch | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|---:|
| No Regularization | 9 | 92.32% | 90.02% | 89.70% |
| Dropout | 10 | 90.49% | 89.02% | 89.19% |
| Weight Decay | 9 | 91.16% | 89.50% | 88.99% |
| Batch Normalization | 9 | 95.57% | 91.13% | 90.91% |

All four configurations achieved more than 88% test accuracy.

## Conclusion

All four LeNet-5 configurations achieved more than 88% test accuracy. Batch Normalization achieved the highest test accuracy at 90.91% and the highest validation accuracy at 91.13%, showing the strongest predictive performance in this experiment. Dropout achieved a test accuracy of 89.19% and had the smallest gap between training and test accuracy, suggesting stronger regularization and less overfitting. Weight Decay achieved 88.99% test accuracy, which was slightly lower than the baseline model's 89.70%. Overall, Batch Normalization provided the highest predictive performance, while Dropout was more effective at reducing the gap between training and test accuracy.

## Repository Files

```text
CS5787-HW1/
├── HW1.ipynb
├── README.md
├── lenet5_convergence.png
├── lenet5_base.pth
├── lenet5_dropout.pth
├── lenet5_weight_decay.pth
└── lenet5_batchnorm.pth

### Hyperparameter Selection

We used a batch size of 64 and a learning rate of 0.001 with the Adam optimizer because these provide a reasonable balance between training stability and computational efficiency for FashionMNIST. We trained all configurations for 10 epochs so that the regularization methods could be compared under the same training conditions.

For Dropout, we used a dropout probability of 0.5, which provides moderate regularization for the fully connected hidden layers. For Weight Decay, we used a coefficient of 1e-4 to provide L2 regularization without excessively restricting the model parameters. The same settings were kept across experiments whenever possible to make the comparison fair.
