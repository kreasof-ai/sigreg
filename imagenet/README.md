# ImageNet Experiments

<img width="2528" height="2528" alt="W B Chart 2_22_2026, 8_22_18 PM" src="https://github.com/user-attachments/assets/9db37ad0-3ab7-4c7c-9973-c40f7248d54e" />

We test SIGReg on [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset with ViT model.

| Model            | Optimizer  | CutMix/MixUp | SIGReg | Epochs | Top-1 Acc |
|------------------|------------|--------------|--------|--------|-----------|
| ViT-Tiny/16      | Muon+AdamW | Yes          | No     | 200    | 69.18%    |
| ViT-Tiny/16      | Muon+AdamW | Yes          | Strong | 200    | 69.28%    |
| ViT-Tiny/16      | Muon+AdamW | Yes          | Weak   | 200    | 72.62%    |

> Note: We use from-scratch pretraining without distillation from larger teacher model.

Comparison with existing technique:

| Model                     | Parameters | Pretraining Data     | Distillation    | Epochs      | Top-1 Acc (%) | Key Reference                                                                                                                               |
| ------------------------- | ---------- | -------------------- | --------------- | ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| ViT-Tiny (original)       | 5.7M       | ImageNet-1K only     | None            | 300         | 72.2          | [A Closer Look at Self-Supervised Lightweight Vision Transformers](https://proceedings.mlr.press/v202/wang23e/wang23e.pdf)                  |
| DeiT-Tiny                 | 5.6M       | ImageNet-1K only     | None            | 300         | 72.2          | [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) |
| ViT-Tiny (enhanced)       | 5.7M       | ImageNet-1K only     | None            | 300         | 74.5          | [A Closer Look at Self-Supervised Lightweight Vision Transformers](https://proceedings.mlr.press/v202/wang23e/wang23e.pdf)                  |
| DeiT-Tiny (distilled)     | 6M         | ImageNet-1K only     | Logit (DeiT)    | 300         | 74.5          | [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf) |
| DeiT-Tiny (ViTKD)         | 5.6M       | ImageNet-1K only     | Feature         | 300–400     | 76.06         | [ViTKD: Feature-based Knowledge Distillation for Vision Transformers](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Yang_ViTKD_Feature-based_Knowledge_Distillation_for_Vision_Transformers_CVPRW_2024_paper.pdf) |
| DeiT-Tiny (manifold)      | 5.6M       | ImageNet-1K only     | Feature (patch) | 300         | 76.5          | [Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation](https://proceedings.nips.cc/paper_files/paper/2022/file/3bd2d73b4e96b0ac5a319be58a96016c-Paper-Conference.pdf) |
| DeiT-Tiny (ViTKD+NKD)     | 6M         | ImageNet-1K only     | Hybrid          | 300–400     | 77.78         | [ViTKD: Practical Guidelines for ViT feature knowledge distillation](https://arxiv.org/abs/2209.02432)  

## Activation Residual Probing

<img width="1389" height="1013" alt="download (2)" src="https://github.com/user-attachments/assets/445a3f98-7df4-4610-a231-090445854a05" />

1. Effective Dimension calculated by Participation Ratio: $d_{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$
    - Weak-SIGReg show higher $d_{eff}$ than Strong-SIGReg and No-SIGReg. While Strong-SIGReg collapse to 1D.
2. Spectral Entropy (Information Capacity): $H = -\sum_i p_i \log p_i$
    - Weak-SIGReg show higher $H$ than Strong-SIGReg and No-SIGReg. With few middle layers almost reach maximum information capacity ceiling for 192 dimension features.
3. Average Pairwise Cosine Similarity (Isotropy): $S = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{a_i \cdot a_j}{\|a_i\| \|a_j\|}$
    - Weak-SIGReg showing noticeably lower average pairwise cosine similarity than No-SIGReg for few middle layers.
    - While Strong-SIGReg showing overall isotropic features. We suspect it's because the model find numerical exploit where a single feature is dominating magnitude to satisfy Strong-SIGReg constraint. Thus, making overall feature distribution isotropic.  
4. L2 Norm Variance (Feature Magnitude): $\sigma^2 = \frac{1}{n} \sum_i (\|a_i\| - \mu)^2$
    - Strong-SIGReg showing significantly higher L2 norm for middle and late layer. While Weak-SIGReg and No-SIGReg show consistent magnitude across layers.