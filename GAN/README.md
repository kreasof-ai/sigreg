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

- Strong SIGReg (alpha_g=0.01, alpha_d=0.1):

- Strong SIGReg (alpha_g=0.05, alpha_d=0.5):