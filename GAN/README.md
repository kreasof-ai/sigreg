# SIGReg for GANs

We investigate the effect of SIGReg on GANs. We found that Strong SIGReg variant can fix the mode collapse problem in GANs while Weak SIGReg doesn't work well.

We use this GAN setup:
- Larger generator than discriminator
- GELU activation function in generator
- LeakyReLU activation function in discriminator
- SIGReg applied both generator and discriminator
- SIGReg applied in every layer
- Alpha SIGReg Generator < Alpha SIGReg Discriminator
- 25 Gaussian synthetic dataset

Result after 400 epochs:
- Baseline:
  <img width="1200" height="1200" alt="heavy_baseline" src="https://github.com/user-attachments/assets/2232baa5-ce7b-4670-bd84-23f9ced454f5" />
- Strong SIGReg (alpha_g=0.01, alpha_d=0.1):
  <img width="1200" height="1200" alt="heavy_strong_1e-1" src="https://github.com/user-attachments/assets/4e8439a1-bae8-44b7-836b-ce0836310b2a" />
- Strong SIGReg (alpha_g=0.05, alpha_d=0.5):
  <img width="1200" height="1200" alt="heavy_strong_5e-1" src="https://github.com/user-attachments/assets/2fbdd22f-f1d5-4b26-98e1-2f3de93be91c" />
