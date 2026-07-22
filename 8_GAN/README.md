# 8. Generative Adversarial Network (GAN)

## Model Overview
A GAN consists of two neural networks — a **Generator** and a **Discriminator** — that compete against each other in a minimax game. The Generator tries to produce fake data indistinguishable from real data, while the Discriminator tries to tell them apart.

## How GANs Work
```
Noise (z) ──► Generator ──► Fake Data ──┐
                                         ├──► Discriminator ──► Real / Fake ?
Real Data ───────────────────────────────┘
```

### Training Loop
1. **Train Discriminator**: Feed real images (label=1) and fake images (label=0). Backpropagate loss.
2. **Train Generator**: Generate fake images, pass through Discriminator, but label them as real (label=1). Generator learns to fool the Discriminator.
3. Repeat until Generator produces realistic samples.

## Dataset
**MNIST Handwritten Digits** (auto-downloaded via Keras)
- 60,000 grayscale images, 28×28 pixels
- Normalized to [-1, 1] range
- Generator learns to create fake handwritten digit images

## Architecture

### Generator
```
Input: Noise vector (latent_dim=100)
Dense(256) → LeakyReLU → BatchNorm
Dense(512) → LeakyReLU → BatchNorm
Dense(1024) → LeakyReLU → BatchNorm
Dense(784, Tanh) → Reshape(28, 28, 1)
```

### Discriminator
```
Input: Image (28×28×1) → Flatten(784)
Dense(1024) → LeakyReLU → Dropout(0.3)
Dense(512)  → LeakyReLU → Dropout(0.3)
Dense(256)  → LeakyReLU → Dropout(0.3)
Dense(1, Sigmoid)
```

## Key Concepts
| Concept | Explanation |
|---------|------------|
| Generator | Creates fake samples from random noise |
| Discriminator | Binary classifier: Real vs Fake |
| Adversarial Loss | Generator maximizes Discriminator error |
| Nash Equilibrium | Training goal: D(x) = 0.5 for all x |
| Mode Collapse | Common failure: Generator produces limited variety |
| Latent Space | The compressed noise vector space (z) |

## Hyperparameters
- **Latent Dimension**: 100
- **Learning Rate**: 0.0002 (Adam, β₁=0.5)
- **Batch Size**: 64
- **Epochs**: 50 (increase for better results)

## Applications of GANs
- 🖼️ **Image Synthesis** — Generate photorealistic faces (StyleGAN)
- 🎨 **Style Transfer** — CycleGAN for painting → photo
- 🏥 **Medical Imaging** — Augment rare disease datasets
- 🎮 **Game Design** — Procedural texture/asset generation
- 🔍 **Super Resolution** — Enhance low-res images (SRGAN)
- 🗣️ **Deepfakes** — Face swapping (misuse case!)

## GAN Variants
| Variant | Use Case |
|---------|----------|
| DCGAN | Deep Convolutional GAN for images |
| cGAN | Conditional GAN (class-conditioned) |
| CycleGAN | Unpaired image-to-image translation |
| StyleGAN | High-quality face generation |
| WGAN | Wasserstein loss for stable training |
| Pix2Pix | Paired image translation |

## How to Run on Google Colab
1. Upload `gan_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **GPU** (recommended for faster training)
3. Runtime → **Run All**
4. Epochs take ~30–60 seconds each on GPU

## Expected Output
- Generator and Discriminator losses printed every 10 epochs
- Grid of generated digit images saved after each epoch
- Final comparison: Real MNIST digits vs GAN-generated digits
- Loss curves showing Generator vs Discriminator training dynamics

## Tips for Better GAN Training
- Use **LeakyReLU** (not ReLU) in Discriminator
- Apply **Label Smoothing** (use 0.9 instead of 1.0 for real labels)
- Use **BatchNormalization** in Generator
- Use **Dropout** in Discriminator
- Don't train one network too far ahead of the other
