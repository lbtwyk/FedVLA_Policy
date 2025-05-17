# FedVLA/DP/test_augmentation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from augmentation import StateAugmenter

def test_state_augmentation():
    """
    Test the state augmentation module with different configurations
    and visualize the results.
    """
    # Create a sample state tensor (batch_size=100, state_dim=7)
    # Representing 6 joint angles + 1 gripper value
    batch_size = 100
    state_dim = 7
    
    # Create a sample state with some pattern
    angles = torch.linspace(-1.0, 1.0, batch_size).unsqueeze(1).repeat(1, 6)
    gripper = torch.zeros(batch_size, 1)
    gripper[:batch_size//2] = 1.0  # Half open, half closed
    
    # Combine into state vector
    state = torch.cat([angles, gripper], dim=1)
    
    # Test different augmentation configurations
    augmenters = [
        # No augmentation (baseline)
        StateAugmenter(enabled=False),
        
        # Gaussian noise with different scales
        StateAugmenter(enabled=True, noise_type="gaussian", noise_scale=0.01),
        StateAugmenter(enabled=True, noise_type="gaussian", noise_scale=0.05),
        
        # Uniform noise
        StateAugmenter(enabled=True, noise_type="uniform", noise_scale=0.05),
        
        # Scaled noise (proportional to state values)
        StateAugmenter(enabled=True, noise_type="scaled", noise_scale=0.1),
        
        # With random dropout
        StateAugmenter(enabled=True, noise_type="gaussian", noise_scale=0.03, random_drop_prob=0.1),
        
        # With clipping
        StateAugmenter(enabled=True, noise_type="gaussian", noise_scale=0.1, clip_bounds=(-0.8, 0.8)),
    ]
    
    # Apply each augmentation and visualize
    plt.figure(figsize=(15, 10))
    
    for i, augmenter in enumerate(augmenters):
        # Apply augmentation
        augmented_state = augmenter(state)
        
        # Plot original vs augmented for the first joint angle
        plt.subplot(len(augmenters), 1, i+1)
        plt.plot(state[:, 0].numpy(), label='Original', alpha=0.7)
        plt.plot(augmented_state[:, 0].numpy(), label='Augmented', alpha=0.7)
        
        # Add title based on configuration
        if not augmenter.enabled:
            title = "No Augmentation"
        else:
            title = f"{augmenter.noise_type.capitalize()} Noise (scale={augmenter.noise_scale})"
            if augmenter.random_drop_prob > 0:
                title += f", Drop Prob={augmenter.random_drop_prob}"
            if augmenter.clip_bounds:
                title += f", Clipped to {augmenter.clip_bounds}"
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("state_augmentation_test.png")
    print("Test visualization saved to 'state_augmentation_test.png'")
    
    # Test noise schedule
    print("\nTesting noise schedule:")
    scheduler = StateAugmenter(
        enabled=True, 
        noise_type="gaussian", 
        noise_scale=0.1,
        noise_schedule="linear_decay"
    )
    
    # Simulate training progress
    epochs = [0, 25, 50, 75, 99]
    total_epochs = 100
    
    for epoch in epochs:
        scheduler.set_training_progress(epoch, total_epochs)
        print(f"Epoch {epoch}/{total_epochs}: Noise scale = {scheduler.noise_scale:.4f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_state_augmentation()
