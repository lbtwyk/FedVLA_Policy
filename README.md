# FedVLA Diffusion Policy

This repository contains the implementation of a diffusion policy model for robot control, based on the FedVLA (Federated Vision-Language-Action) framework. The diffusion policy model learns to predict robot actions (joint angles and gripper state) from visual observations using a denoising diffusion probabilistic model (DDPM) approach.

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
│   ├── inference.py        # Inference script
│   ├── run.sh              # Training script wrapper
│   └── checkpoints/        # Model checkpoints (created during training)
└── mycobot_episodes/       # Dataset directory (collection of robot episodes)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- tqdm
- Pillow

Install the required packages:

```bash
pip install torch torchvision tqdm pillow
```

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
