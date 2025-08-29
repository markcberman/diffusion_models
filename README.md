# DDIM Diffusion Models for CIFAR-10

This repository contains a PyTorch implementation of DDIM (Denoising Diffusion Implicit Models) trained on the CIFAR-10 dataset.

## Overview

DDIM is a variant of diffusion models that enables faster sampling through deterministic generation while maintaining high-quality image synthesis. This implementation includes:

- Complete DDIM training pipeline
- Custom UNet architecture with time embeddings
- CIFAR-10 dataset integration
- Training and sampling utilities

## Features

- **Batch Size**: 32
- **Training**: Configurable epochs and iterations per epoch (defaults: 1 epoch, 1 iteration)
- **Model Architecture**: UNet with skip connections and time conditioning
- **Sampling**: Deterministic DDIM sampling (eta=0.0)
- **Image Size**: 32x32 (CIFAR-10 native resolution)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the DDIM model on CIFAR-10:

```bash
# Use default settings (1 epoch, 1 iteration - for quick testing)
python ddim_cifar.py

# Customize training parameters
python ddim_cifar.py --epochs 50 --iterations 1000

# Train for 10 epochs with 500 iterations each
python ddim_cifar.py --epochs 10 --iterations 500
```

**Command Line Arguments:**
- `--epochs`: Number of training epochs (default: 1)
- `--iterations`: Number of iterations per epoch (default: 1)

The training process will:
- Download CIFAR-10 dataset automatically
- Train for the specified number of epochs with the specified iterations each
- Save model checkpoints every 10 epochs (if training for 10+ epochs)
- Generate sample images after training

### Model Checkpoints

Model checkpoints are saved as:
- `ddim_model_epoch_X.pth` - Every 10 epochs
- `ddim_model_final.pth` - Final trained model

### Sampling

After training, the model will automatically generate 16 sample images and save them as `ddim_samples.png`.

## Architecture

- **UNet**: Encoder-decoder with skip connections
- **Time Embeddings**: Sinusoidal positional encodings
- **Blocks**: Downsampling/upsampling with time conditioning
- **DDIM Process**: Deterministic reverse diffusion

## Results

The trained model generates high-quality 32x32 images from random noise through the DDIM sampling process. The deterministic nature (eta=0.0) provides faster and more stable generation compared to traditional DDPM sampling.

## License

This project is open source and available under the MIT License.
