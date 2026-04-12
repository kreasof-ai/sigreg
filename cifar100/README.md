# Results CIFAR-100

| Model    | Optimizer  | CutMix/MixUp | SIGReg | Epochs | Top-1 Acc |
|----------|------------|--------------|--------|--------|-----------|
| ResNet18 | SGD        | No           | No     | 1600   | 79.03%    |
| ResNet18 | SGD        | No           | Strong | 1600   | 78.86%    |
| ResNet18 | SGD        | No           | Weak   | 1600   | 79.42%    |
| ResNet18 | SGD        | Yes          | No     | 400    | 82.13%    |
| ResNet18 | SGD        | Yes          | Strong | 400    | 81.18%    |
| ResNet18 | SGD        | Yes          | Weak   | 400    | 82.13%    |
| ViT      | AdamW      | Yes          | No     | 400    | 20.73%    |
| ViT      | AdamW      | Yes          | Strong | 400    | 70.20%    |
| ViT      | AdamW      | Yes          | Weak   | 400    | 72.02%    |
| ViT      | Muon+AdamW | No           | No     | 400    | 58.77%    |
| ViT      | Muon+AdamW | No           | Strong | 400    | 63.16%    |
| ViT      | Muon+AdamW | No           | Weak   | 400    | 67.52%    |
| ViT      | Muon+AdamW | Yes          | No     | 400    | 62.44%    |
| ViT      | Muon+AdamW | Yes          | Strong | 400    | 74.34%    |
| ViT      | Muon+AdamW | Yes          | Weak   | 400    | 74.56%    |

## Interesting Findings

One of the nastiest comparison happen between sample with this setup:
- ViT (Vision Transformer)
- AdamW optimizer
- Heavy data augmentation (RandomCrop, RandomHorizontalFlip, RandAugment, CutMix, and MixUp)

Top-1 accuracy on CIFAR-100:
- Without SIGReg: 20.73%
- With Strong SIGReg: 70.20%
- With Weak SIGReg: 72.02%

Loss curve on CIFAR-100:
- Without SIGReg: 
    ![without SIGReg loss](cifar100/plot/CIFAR_100_SIGReg_Baseline_ViT_CutMix_MixUp_AdamW_loss.png)
- With Strong SIGReg: 
    ![with strong SIGReg loss](cifar100/plot/CIFAR_100_SIGReg_Strong_ViT_CutMix_MixUp_AdamW_loss.png)
- With Weak SIGReg: 
    ![with weak SIGReg loss](cifar100/plot/CIFAR_100_SIGReg_Weak_ViT_CutMix_MixUp_AdamW_loss.png)

## Fixing ViT Baseline

Because we find ViT + AdamW baseline perform very poorly, we test it on specific setup to ensure peak baseline performance. Few intervention we choose are:
- Using specific model shape.
- Using different weight decay.
- Using absolute position embedding, not per layer 2D RoPE.
- Using specific initialization.
- Using gradient clipping.
- Using LR Scheduler eta_min = 1e-5.
- Using different mean and std for CIFAR-100 dataset.
- Using drop path rate of 0.1 without dropout.
- Write manual attention forward pass.
- Using QKV bias.

The results after ViT intervention:

| Model    | Optimizer  | CutMix/MixUp | SIGReg | Epochs | Top-1 Acc |
|----------|------------|--------------|--------|--------|-----------|
| ViT      | AdamW      | Yes          | No     | 400    | 70.76%    |
| ViT      | AdamW      | Yes          | Strong | 400    | 72.71%    |
| ViT      | AdamW      | Yes          | Weak   | 400    | 71.65%    |
| ViT      | Muon+AdamW | Yes          | No     | 400    | 75.87%    |
| ViT      | Muon+AdamW | Yes          | Strong | 400    | 76.98%    |
| ViT      | Muon+AdamW | Yes          | Weak   | 400    | 76.24%    |

## Vanilla MLP Experiments

We test vanilla MLP on CIFAR-100 dataset to show the effect of SIGReg on simple model. We use MLP with this setup:
- 6 layers deep.
- Direct flattened input from image.
- 1024 hidden dimension.
- ReLU activation.
- No dropout.
- No batch norm.
- No residual connection.
- Pure SGD optimizer without momentum.
- No LR Scheduler.
- No weight decay.

| Model    | Optimizer  | CutMix/MixUp | SIGReg | Epochs | Top-1 Acc |
|----------|------------|--------------|--------|--------|-----------|
| MLP      | SGD        | No           | No     | 400    | 26.77%    |
| MLP      | SGD        | No           | Strong | 400    | 35.99%    |
| MLP      | SGD        | No           | Weak   | 400    | 42.17%    |
| MLP      | SGD        | Yes          | No     | 400    | 38.08%    |
| MLP      | SGD        | Yes          | Strong | 400    | 38.70%    |
| MLP      | SGD        | Yes          | Weak   | 400    | 38.40%    |

The possible explanation why CutMix/MixUp version perform worse for weak SIGReg is that 400 epochs is not enough to converge into higher ceiling because the last recorded loss is still decreasing, since this is flat LR schedule. While no CutMix/MixUp version can converge much faster.

> Note: MLP experiments mistakenly still uses gradient clipping, we will test it without gradient clipping in the future.