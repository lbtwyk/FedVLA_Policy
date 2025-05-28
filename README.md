# FedVLA Diffusion Policy

This repository contains the implementation of a diffusion policy model for robot control, based on the FedVLA (Federated Vision-Language-Action) framework. The diffusion policy model learns to predict robot actions (joint angles and gripper state) from visual observations using a denoising diffusion probabilistic model (DDPM) approach.

## Demo

Watch the FedVLA inference in action:

[🎥 View Demo Video](https://liveuclac-my.sharepoint.com/:v:/g/personal/zcabaax_ucl_ac_uk/EYtNW_m0uf9CknEi12tfdVcBMjQBGibRLapVYdBSpMbGwA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=HV1eUH)

*The demo shows the diffusion policy model performing real-time robot control with visual feedback.*

## Overview

The diffusion policy model is a conditional generative model that:
1. Takes an image observation as input
2. Generates a robot action (joint angles + gripper state)
3. Uses a denoising diffusion process for robust action generation

The model architecture consists of:
- A ResNet-34 backbone for image feature extraction
- A diffusion model that predicts noise in the action space
- A time-step embedding module for the diffusion process

## Repository Structure

```
.
├── DP/                     # Diffusion Policy implementation
│   ├── train.py            # Training script
│   ├── model.py            # Model architecture
│   ├── dataset.py          # Dataset loader
│   ├── augmentation.py     # State augmentation for robustness
│   ├── inference.py        # Inference script
│   ├── run.sh              # Training script wrapper
│   └── checkpoints/        # Model checkpoints (created during training)
└── mycobot_episodes/       # Dataset directory (collection of robot episodes)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with appropriate CUDA or MPS support)
- torchvision
- matplotlib
- numpy
- tqdm
- Pillow

## Setup Instructions

### Mac Setup (Apple Silicon or Intel)

1. **Create a virtual environment**:
   ```bash
   python -m venv dp_venv
   source dp_venv/bin/activate
   ```

2. **Install PyTorch with MPS support** (for Apple Silicon):
   ```bash
   pip install --upgrade pip
   pip install torch torchvision
   ```
   This will install PyTorch with MPS (Metal Performance Shaders) acceleration for Apple Silicon Macs.

3. **Install other dependencies**:
   ```bash
   pip install matplotlib numpy tqdm pillow
   ```

4. **Verify MPS availability** (for Apple Silicon):
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   print(f"MPS built: {torch.backends.mps.is_built()}")
   ```

5. **Troubleshooting MPS issues**:
   - If you encounter MPS-related errors, try setting these environment variables:
     ```bash
     export PYTORCH_ENABLE_MPS_FALLBACK=1
     ```
   - For specific operations not supported by MPS, the model will automatically fall back to CPU

### Windows Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv dp_venv
   dp_venv\Scripts\activate
   ```

2. **Install PyTorch with CUDA support** (for NVIDIA GPUs):
   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   Replace `cu118` with your desired CUDA version (e.g., `cu117`, `cu121`).

3. **Install other dependencies**:
   ```bash
   pip install matplotlib numpy tqdm pillow
   ```

4. **Verify CUDA availability**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")
   print(f"Current device: {torch.cuda.current_device()}")
   print(f"Device name: {torch.cuda.get_device_name(0)}")
   ```

### CPU-only Setup (Any Platform)

If you don't have a compatible GPU:

```bash
pip install torch torchvision
pip install matplotlib numpy tqdm pillow
```

Note that training will be significantly slower without GPU acceleration.

## Dataset Structure

The dataset follows a specific structure:

```
mycobot_episodes/
├── collection_YYYYMMDD_HHMMSS/
│   └── episode_XXX/
│       ├── states.json
│       └── frame_dir/
│           ├── image_0000.png
│           ├── image_0001.png
│           └── ...
└── ...
```

Each `states.json` file contains a list of timesteps, where each timestep has:
- `angles`: List of 6 joint angles
- `gripper_value`: List containing a single gripper value
- `image`: Path to the corresponding image file (relative to the episode directory)

## Training

To train the diffusion policy model:

```bash
cd DP
bash run.sh
```

The `run.sh` script calls `train.py` with the following default parameters:

```bash
python train.py --data_dir ../mycobot_episodes/ --output_dir ./checkpoints --num_epochs 801 --batch_size 64 --save_interval 50 --eval_interval 50
```

### Training Parameters

Key parameters for training:

- `--data_dir`: Path to the dataset directory
- `--output_dir`: Directory to save model checkpoints
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--save_interval`: Save model checkpoint every N epochs
- `--eval_interval`: Evaluate model every N epochs
- `--state_dim`: Dimension of the state vector (default: 7 - 6 joint angles + 1 gripper value)
- `--diffusion_timesteps`: Number of diffusion timesteps (default: 1000)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)

### State Augmentation

The model supports state augmentation to improve robustness and generalization:

```bash
python train.py --data_dir ../mycobot_episodes/ --state_aug_enabled --state_aug_noise_type gaussian --state_aug_noise_scale 0.05
```

State augmentation parameters:

- `--state_aug_enabled`: Enable state augmentation during training
- `--state_aug_noise_type`: Type of noise to apply (`gaussian`, `uniform`, or `scaled`)
- `--state_aug_noise_scale`: Scale of noise to apply (e.g., 0.01 to 0.1)
- `--state_aug_noise_schedule`: How noise scale changes over training (`constant`, `linear_decay`, or `cosine_decay`)
- `--state_aug_random_drop_prob`: Probability of randomly zeroing out a joint value (0.0 to disable)
- `--state_aug_clip_min` and `--state_aug_clip_max`: Min/max values to clip augmented state

### Early Stopping

To prevent overfitting, you can enable early stopping:

```bash
python train.py --data_dir ../mycobot_episodes/ --early_stopping --patience 15 --restore_best_weights
```

Early stopping parameters:

- `--early_stopping`: Enable early stopping based on validation performance
- `--patience`: Number of evaluations to wait for improvement before stopping
- `--min_delta`: Minimum change in validation metric to qualify as improvement
- `--restore_best_weights`: Restore model to best weights when early stopping occurs

For a complete list of parameters, see the argument parser in `train.py`.

## Model Architecture

The diffusion policy model consists of:

1. **Image Encoder**: ResNet-34 backbone to extract visual features
2. **Time Embedding**: Sinusoidal positional embedding for diffusion timesteps
3. **MLP Layers**: Multiple MLP blocks that process the combined state and image features
4. **Output Layer**: Projects to the state dimension to predict noise

The model is trained to predict the noise added during the forward diffusion process, which is then used during sampling to generate actions.

## Inference

To run inference with a trained model:

```bash
python inference.py --checkpoint_path ./checkpoints/model_best.pth --image_path path/to/test/image.png
```

## Evaluation

The model is evaluated by:
1. Sampling actions from the diffusion model
2. Comparing the sampled actions to ground truth actions
3. Computing the Mean Squared Error (MSE) between predicted and ground truth actions

## License

[MIT License](LICENSE)

## Acknowledgements

This project builds upon the following works:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
