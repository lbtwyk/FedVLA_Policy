# FedVLA/DP/augmentation.py

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Callable

class StateAugmenter:
    """
    A class for augmenting robot state vectors (joint angles and gripper values)
    during training to improve policy robustness and generalization.
    
    This augmenter applies controlled noise or perturbations to the state inputs
    with configurable parameters.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        noise_type: str = "gaussian",
        noise_scale: float = 0.01,
        noise_schedule: str = "constant",
        clip_bounds: Optional[Tuple[float, float]] = None,
        joint_mask: Optional[List[bool]] = None,
        random_drop_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the state augmenter.
        
        Args:
            enabled (bool): Whether augmentation is enabled.
            noise_type (str): Type of noise to apply. Options: "gaussian", "uniform", "scaled".
            noise_scale (float): Scale of the noise to apply.
            noise_schedule (str): How noise scale changes over training.
                Options: "constant", "linear_decay", "cosine_decay".
            clip_bounds (Tuple[float, float], optional): Min and max values to clip augmented state.
            joint_mask (List[bool], optional): Which joints to apply augmentation to.
                If None, applies to all joints.
            random_drop_prob (float): Probability of randomly zeroing out a joint value (simulating sensor dropout).
            seed (int, optional): Random seed for reproducibility.
        """
        self.enabled = enabled
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.base_noise_scale = noise_scale  # Store original scale for schedules
        self.noise_schedule = noise_schedule
        self.clip_bounds = clip_bounds
        self.random_drop_prob = random_drop_prob
        
        # Default joint mask applies to all joints
        if joint_mask is None:
            self.joint_mask = None  # Will be set based on input tensor shape
        else:
            self.joint_mask = torch.tensor(joint_mask, dtype=torch.bool)
            
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Training progress tracking for schedules
        self.current_epoch = 0
        self.total_epochs = 1  # Will be updated when set_training_progress is called
        
    def set_training_progress(self, current_epoch: int, total_epochs: int):
        """
        Update the training progress for noise schedules.
        
        Args:
            current_epoch (int): Current training epoch.
            total_epochs (int): Total number of training epochs.
        """
        self.current_epoch = current_epoch
        self.total_epochs = max(1, total_epochs)  # Ensure at least 1 to avoid division by zero
        
        # Update noise scale based on schedule
        progress = self.current_epoch / self.total_epochs
        
        if self.noise_schedule == "linear_decay":
            # Linear decay from base_noise_scale to 0
            self.noise_scale = self.base_noise_scale * (1 - progress)
        elif self.noise_schedule == "cosine_decay":
            # Cosine decay from base_noise_scale to 0
            self.noise_scale = self.base_noise_scale * 0.5 * (1 + np.cos(np.pi * progress))
        # "constant" schedule doesn't change the noise scale
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to the state tensor.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim).
                For robot control, typically contains joint angles and gripper values.
                
        Returns:
            torch.Tensor: Augmented state tensor of the same shape.
        """
        if not self.enabled or self.noise_scale <= 0:
            return state
            
        # Create a copy to avoid modifying the original tensor
        augmented_state = state.clone()
        
        # Create or adjust joint mask if needed
        if self.joint_mask is None:
            # Apply to all dimensions
            mask = torch.ones_like(augmented_state, dtype=torch.bool)
        else:
            # Expand mask to match batch dimension
            mask = self.joint_mask.expand_as(augmented_state)
        
        # Apply random dropout (zeroing) if enabled
        if self.random_drop_prob > 0:
            dropout_mask = torch.rand_like(augmented_state) > self.random_drop_prob
            mask = mask & dropout_mask
        
        # Generate noise based on selected type
        if self.noise_type == "gaussian":
            noise = torch.randn_like(augmented_state) * self.noise_scale
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(augmented_state) * 2 - 1) * self.noise_scale
        elif self.noise_type == "scaled":
            # Scale noise proportionally to the absolute value of each state component
            # This applies more noise to larger values and less to smaller values
            abs_scale = torch.abs(augmented_state).clamp(min=1e-6)
            noise = torch.randn_like(augmented_state) * self.noise_scale * abs_scale
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Apply noise only to masked elements
        augmented_state = torch.where(mask, augmented_state + noise, augmented_state)
        
        # Clip values if bounds are provided
        if self.clip_bounds is not None:
            min_val, max_val = self.clip_bounds
            augmented_state = torch.clamp(augmented_state, min_val, max_val)
            
        return augmented_state

# --- Example Usage ---
if __name__ == "__main__":
    # Example state tensor (batch_size=2, state_dim=7)
    # Representing 6 joint angles + 1 gripper value
    state = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]
    ], dtype=torch.float32)
    
    # Create augmenter with default parameters
    augmenter = StateAugmenter(
        enabled=True,
        noise_type="gaussian",
        noise_scale=0.02,
        clip_bounds=(-1.0, 1.0)
    )
    
    # Apply augmentation
    augmented_state = augmenter(state)
    
    print("Original state:")
    print(state)
    print("\nAugmented state:")
    print(augmented_state)
    print("\nDifference:")
    print(augmented_state - state)
