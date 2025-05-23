# FedVLA/DP/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import logging
from typing import Optional, Tuple

# Configure logging if run as main, otherwise assume it's configured elsewhere
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal positional embeddings for the diffusion timestep.
    Taken from: https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    def __init__(self, dim: int):
        """
        Initializes the sinusoidal positional embedding module.

        Args:
            dim (int): The dimension of the embedding space.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal positional embeddings for given timesteps.

        Args:
            time (torch.Tensor): Timesteps tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionPolicyModel(nn.Module):
    """
    Diffusion-based policy model that predicts noise to denoise states conditioned on images.
    Uses ResNet-34 for image feature extraction and MLPs for state processing.
    """
    
    def __init__(self, state_dim: int = 7, time_emb_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 4, image_feature_dim: int = 512,
                 use_pretrained_resnet: bool = True, freeze_resnet: bool = True):
        """
        Initializes the Diffusion Policy Model.

        Args:
            state_dim (int): Dimension of the robot state vector (e.g., joint positions).
            time_emb_dim (int): Dimension of the timestep embedding.
            hidden_dim (int): Hidden dimension for MLP layers.
            num_layers (int): Number of MLP layers.
            image_feature_dim (int): Feature dimension from ResNet backbone (512 for ResNet-34).
            use_pretrained_resnet (bool): Whether to use pretrained ResNet weights.
            freeze_resnet (bool): Whether to freeze ResNet parameters.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.image_feature_dim = image_feature_dim
        
        # Timestep embedding using sinusoidal positional encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Image feature extractor using ResNet-34
        if use_pretrained_resnet:
            self.image_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            logging.info("Using pretrained ResNet-34 weights")
        else:
            self.image_encoder = models.resnet34(weights=None)
            logging.info("Using randomly initialized ResNet-34 weights")
            
        # Remove the final classification layer and replace with identity
        self.image_encoder.fc = nn.Identity()
        
        # Freeze ResNet parameters if specified
        if freeze_resnet:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            logging.info("ResNet-34 parameters frozen")
        else:
            logging.info("ResNet-34 parameters will be fine-tuned")
        
        # Combined feature dimension: state + time_embedding + image_features
        combined_dim = state_dim + time_emb_dim + image_feature_dim
        
        # MLP layers for processing combined features
        layers = []
        input_dim = combined_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Mish(),  # Mish activation often works well for diffusion models
                nn.Dropout(0.1)  # Light dropout for regularization
            ])
            input_dim = hidden_dim
        
        # Final output layer to predict noise (same dimension as state)
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logging.info(f"DiffusionPolicyModel initialized:")
        logging.info(f"  - State dim: {state_dim}")
        logging.info(f"  - Time embedding dim: {time_emb_dim}")
        logging.info(f"  - Image feature dim: {image_feature_dim} (ResNet-34)")
        logging.info(f"  - Hidden dim: {hidden_dim}")
        logging.info(f"  - MLP layers: {num_layers}")
        logging.info(f"  - Combined input dim: {combined_dim}")
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, timestep: torch.Tensor, 
                image_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion policy model.

        Args:
            state (torch.Tensor): Noisy state tensor of shape (batch_size, state_dim).
            timestep (torch.Tensor): Timestep tensor of shape (batch_size,).
            image_input (torch.Tensor): Image tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Predicted noise tensor of shape (batch_size, state_dim).
        """
        batch_size = state.shape[0]
        
        # Generate timestep embeddings
        time_emb = self.time_mlp(timestep.float())  # Shape: (batch_size, time_emb_dim)
        
        # Extract image features using ResNet-34
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.image_encoder.parameters())):
            image_features = self.image_encoder(image_input)  # Shape: (batch_size, 512)
        
        # Combine all features
        combined_features = torch.cat([state, time_emb, image_features], dim=1)  
        # Shape: (batch_size, state_dim + time_emb_dim + 512)
        
        # Process through MLP to predict noise
        predicted_noise = self.mlp(combined_features)  # Shape: (batch_size, state_dim)
        
        return predicted_noise

# --- Test/Demo Code ---
if __name__ == "__main__":
    # Test the model with ResNet-34
    print("Testing DiffusionPolicyModel with ResNet-34...")
    
    # Model parameters
    batch_size = 4
    state_dim = 7
    image_height, image_width = 224, 224
    
    # Create model
    model = DiffusionPolicyModel(
        state_dim=state_dim,
        time_emb_dim=64,
        hidden_dim=256,
        num_layers=4,
        image_feature_dim=512,  # ResNet-34 feature dimension
        use_pretrained_resnet=True,
        freeze_resnet=True
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create sample inputs
    state = torch.randn(batch_size, state_dim)
    timestep = torch.randint(0, 1000, (batch_size,))
    image = torch.randn(batch_size, 3, image_height, image_width)
    
    print(f"\nInput shapes:")
    print(f"  State: {state.shape}")
    print(f"  Timestep: {timestep.shape}")
    print(f"  Image: {image.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(state, timestep, image)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {state_dim})")
    print(f"✓ Forward pass successful!")
    
    # Test that output has correct properties
    assert output.shape == (batch_size, state_dim), f"Output shape mismatch: {output.shape} vs ({batch_size}, {state_dim})"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"
    
    print("✓ All tests passed!")
