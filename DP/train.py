# FedVLA/DP/train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import argparse
import math
from typing import Optional, List, Tuple, Union, Dict

# Import custom modules
from dataset import RobotEpisodeDataset # Assuming dataset.py is in the same directory
from model import DiffusionPolicyModel  # Assuming model.py is in the same directory
from augmentation import StateAugmenter  # Import the state augmentation module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_mps_cache():
    """
    Cleans the MPS cache to prevent memory explosion on Apple Silicon hardware.
    This function should be called periodically during long training runs.
    """
    # Check if MPS is available
    if torch.backends.mps.is_available():
        # Force garbage collection
        import gc
        gc.collect()

        # Empty MPS cache
        torch.mps.empty_cache()
        return True
    return False

# --- Diffusion Schedule Helpers ---

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Generates a linear schedule for beta values."""
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extracts the appropriate schedule values for a batch of timesteps t."""
    batch_size = t.shape[0]
    # Ensure t has the same device as a for gather
    out = a.to(t.device).gather(-1, t)
    # Reshape for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Forward Diffusion Process (Adding Noise) ---

def q_sample(x_start: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Adds noise to the data x_start according to the timestep t."""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_x

# --- Reverse Diffusion Process (Sampling/Denoising) ---

@torch.no_grad() # Sampling doesn't require gradients
def p_sample(model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int,
             betas: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor,
             sqrt_recip_alphas: torch.Tensor, posterior_variance: torch.Tensor,
             image_input: torch.Tensor) -> torch.Tensor:
    """
    Performs one step of the DDPM reverse process (sampling).
    x_{t-1} ~ p(x_{t-1} | x_t)

    Args:
        model: The diffusion model.
        x (torch.Tensor): The noisy state at timestep t (x_t), shape (batch_size, state_dim).
        t (torch.Tensor): The current timestep t for the batch, shape (batch_size,).
        t_index (int): The integer index corresponding to timestep t.
        betas: Precomputed schedule tensor.
        sqrt_one_minus_alphas_cumprod: Precomputed schedule tensor.
        sqrt_recip_alphas: Precomputed schedule tensor.
        posterior_variance: Precomputed schedule tensor.
        image_input (torch.Tensor): The conditioning image input, shape (batch_size, C, H, W).

    Returns:
        torch.Tensor: The estimated state at timestep t-1 (x_{t-1}).
    """
    # Use model to predict noise added at step t
    predicted_noise = model(state=x, timestep=t, image_input=image_input)

    # Get schedule values for timestep t
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Calculate the mean of the posterior p(x_{t-1} | x_t, x_0)
    # This is the core DDPM sampling equation
    mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        # If t=0, the sample is deterministic (no noise added)
        return mean
    else:
        # If t > 0, add noise based on the posterior variance
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4: x_{t-1} = mean + sqrt(posterior_variance) * noise
        return mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: tuple, timesteps: int,
                  betas: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor,
                  sqrt_recip_alphas: torch.Tensor, posterior_variance: torch.Tensor,
                  device: torch.device, image_input: torch.Tensor) -> torch.Tensor:
    """
    Performs the full DDPM sampling loop, starting from noise.

    Args:
        model: The diffusion model.
        shape (tuple): The desired shape of the output tensor (batch_size, state_dim).
        timesteps (int): Total number of diffusion steps.
        betas, ...: Precomputed schedule tensors.
        device: The device to perform sampling on.
        image_input (torch.Tensor): The conditioning image input for the batch.

    Returns:
        torch.Tensor: The final denoised sample (predicted x_0).
    """
    batch_size = shape[0]

    # Start from pure noise (x_T)
    img = torch.randn(shape, device=device)

    # Iterate backwards from T-1 down to 0
    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling", total=timesteps, leave=False):
        # Create timestep tensor for the current index i
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        # Perform one denoising step
        img = p_sample(model, img, t, i,
                       betas, sqrt_one_minus_alphas_cumprod,
                       sqrt_recip_alphas, posterior_variance,
                       image_input) # Pass conditioning image

        # For MPS device, synchronize to ensure operations are complete
        if device.type == "mps":
            torch.mps.synchronize()

            # Periodically clean MPS cache during long sampling processes
            # Clean every 100 steps to avoid too frequent cleaning
            if i % 100 == 0 and i > 0:
                clean_mps_cache()

    # img now holds the predicted x_0

    # Final synchronization for MPS device
    if device.type == "mps":
        torch.mps.synchronize()
        # Optional: clear cache to free up memory
        torch.mps.empty_cache()

    return img


# --- Custom Collate Function ---
# (Keep the existing custom_collate_fn as is)
def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
    """
    Custom collate function to handle batching of (state, image_tensor) tuples.
    Filters out invalid items and returns (None, None) if the resulting batch would be empty.
    """
    states = []
    images = []
    valid_item_count = 0
    for i, item in enumerate(batch):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            # logging.warning(f"Skipping malformed batch item at index {i}: Type {type(item)}, Value {item}")
            continue
        state, image = item
        if not (isinstance(state, torch.Tensor) and isinstance(image, torch.Tensor)):
             # logging.warning(f"Skipping batch item at index {i} with non-tensor element: state type {type(state)}, image type {type(image)}")
             continue
        states.append(state)
        images.append(image)
        valid_item_count += 1

    if valid_item_count == 0:
        if batch:
             logging.warning(f"Collate function resulted in an empty batch after filtering {len(batch)} items. Check dataset __getitem__ for errors.")
        return None, None

    try:
        states_batch = torch.stack(states, dim=0)
        images_batch = torch.stack(images, dim=0)
    except Exception as e:
         logging.error(f"Error stacking {valid_item_count} valid tensors in collate_fn: {e}. ")
         for i in range(min(3, len(states))):
             logging.error(f"  Sample {i}: state shape {states[i].shape}, image shape {images[i].shape}")
         return None, None

    return states_batch, images_batch

# --- Evaluation Function (Modified for Sampling) ---

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
             diffusion_timesteps: int, betas: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor, sqrt_recip_alphas: torch.Tensor,
             posterior_variance: torch.Tensor, num_eval_samples: int) -> float:
    """
    Evaluates the model by sampling predictions and comparing to ground truth.

    Args:
        model: The diffusion policy model.
        dataloader: DataLoader for the evaluation dataset.
        device: The device to run evaluation on (CPU or GPU).
        diffusion_timesteps, betas, ...: Schedule tensors needed for sampling.
        num_eval_samples (int): The number of samples to generate and evaluate.

    Returns:
        Average Mean Squared Error (MSE) between predicted and ground truth states.
    """
    model.eval() # Set model to evaluation mode
    total_mse = 0.0
    samples_evaluated = 0

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(dataloader, desc="Evaluating (Sampling)", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            if samples_evaluated >= num_eval_samples:
                break # Stop after evaluating the desired number of samples

            if batch is None or batch == (None, None):
                logging.warning(f"Skipping empty/invalid batch during evaluation at index {batch_idx}.")
                continue

            try:
                gt_state_batch, image_batch = batch # Ground Truth state
            except Exception as e:
                 logging.error(f"Error unpacking evaluation batch {batch_idx}: {e}")
                 continue

            if gt_state_batch is None or image_batch is None:
                 logging.warning(f"Skipping evaluation batch {batch_idx} due to None tensor.")
                 continue

            # Determine how many samples from this batch to evaluate
            batch_size = gt_state_batch.shape[0]
            samples_to_take = min(batch_size, num_eval_samples - samples_evaluated)
            if samples_to_take <= 0: continue # Should not happen with outer check, but safety

            # Select subset of the batch if needed
            gt_state_batch = gt_state_batch[:samples_to_take].to(device)
            image_batch = image_batch[:samples_to_take].to(device)

            if gt_state_batch.shape[0] == 0 or image_batch.shape[0] == 0:
                 logging.warning(f"Skipping empty evaluation batch {batch_idx} after device transfer/subsetting.")
                 continue

            # --- Perform Sampling ---
            logging.info(f"Generating {gt_state_batch.shape[0]} samples for evaluation batch {batch_idx}...")
            predicted_state_batch = p_sample_loop(
                model,
                shape=gt_state_batch.shape, # Shape of the state tensor
                timesteps=diffusion_timesteps,
                betas=betas,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas=sqrt_recip_alphas,
                posterior_variance=posterior_variance,
                device=device,
                image_input=image_batch # Pass the conditioning image
            )
            logging.info(f"Sample generation finished for batch {batch_idx}.")


            # --- Calculate Metric (MSE) ---
            mse = F.mse_loss(predicted_state_batch, gt_state_batch, reduction='sum') # Sum MSE over batch
            total_mse += mse.item()
            samples_evaluated += gt_state_batch.shape[0]

            # Log comparison for the first sample in the batch
            if batch_idx == 0 and samples_evaluated > 0:
                 logging.info("--- Evaluation Sample Comparison (First Batch) ---")
                 logging.info(f"Ground Truth State (first sample): {gt_state_batch[0].cpu().numpy()}")
                 logging.info(f"Predicted State (first sample):  {predicted_state_batch[0].cpu().numpy()}")
                 logging.info("-------------------------------------------------")

            progress_bar.set_postfix(samples=f"{samples_evaluated}/{num_eval_samples}")


    model.train() # Set model back to training mode

    if samples_evaluated == 0:
        logging.warning("Evaluation sampling completed without evaluating any samples.")
        return float('inf') # Or handle as appropriate

    avg_mse = total_mse / samples_evaluated # Calculate average MSE per sample
    return avg_mse


# --- Training Function ---

def train(args):
    """
    Main training loop for the diffusion policy model.
    """
    # --- Device Setup ---
    # Check for MPS (Metal Performance Shaders) availability for Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Metal)")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logging.info(f"Using CUDA device {args.gpu_id}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    logging.info(f"Device: {device}")

    # --- Diffusion Schedule Setup ---
    timesteps = args.diffusion_timesteps
    betas = linear_beta_schedule(timesteps=timesteps, beta_start=args.beta_start, beta_end=args.beta_end).to(device) # Move schedule to device
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # Precompute values needed for q_sample AND p_sample
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    # Calculate posterior variance q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    # Clip variance to avoid issues at t=0 (though p_sample handles t=0 separately)
    # posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)

    logging.info(f"Diffusion schedule set up with {timesteps} timesteps.")

    # --- Transforms ---
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset Loading and Splitting ---
    try:
        import inspect
        from torch.utils.data import random_split

        sig = inspect.signature(RobotEpisodeDataset.__init__)

        # Determine if we should use a separate validation dataset or split a single dataset
        use_separate_val_dataset = args.eval_data_dir is not None and args.eval_data_dir != args.data_dir and args.eval_data_dir != ""

        if use_separate_val_dataset:
            # --- Load Training Dataset ---
            train_dataset_args = {
                'base_dir': args.data_dir,
                'num_episodes': args.num_episodes,
                'flat_structure': True  # Use flat structure as default
            }
            if 'transform' in sig.parameters: train_dataset_args['transform'] = image_transform

            logging.info(f"Loading training dataset from {args.data_dir} with flat structure")
            train_dataset = RobotEpisodeDataset(**train_dataset_args)

            if len(train_dataset) == 0:
                logging.error("Training dataset is empty.");
                return

            # --- Load Separate Validation Dataset ---
            if args.eval_interval > 0:
                eval_dataset_args = {
                    'base_dir': args.eval_data_dir,
                    'num_episodes': args.eval_num_episodes if args.eval_num_episodes else args.num_episodes,
                    'flat_structure': True  # Use flat structure as default
                }
                if 'transform' in sig.parameters: eval_dataset_args['transform'] = image_transform

                logging.info(f"Loading evaluation dataset from {args.eval_data_dir} with flat structure")
                eval_dataset = RobotEpisodeDataset(**eval_dataset_args)

                if eval_dataset is None or len(eval_dataset) == 0:
                    logging.warning("Evaluation dataset is empty or failed to load. Skipping evaluation.")
                    eval_dataset = None
                    eval_dataloader = None
            else:
                logging.info("Evaluation interval is 0. Skipping evaluation setup.")
                eval_dataset = None
                eval_dataloader = None
        else:
            # --- Load Single Dataset and Split ---
            # Use flat structure as default
            logging.info(f"Using single dataset with train/val split (ratio: {args.val_split_ratio})")

            full_dataset_args = {
                'base_dir': args.data_dir,
                'num_episodes': 0,  # Load all episodes
                'flat_structure': True  # Use flat structure as default
            }
            if 'transform' in sig.parameters: full_dataset_args['transform'] = image_transform

            logging.info(f"Loading full dataset from {args.data_dir} with flat structure for train/val split")
            full_dataset = RobotEpisodeDataset(**full_dataset_args)

            if len(full_dataset) == 0:
                logging.error("Dataset is empty.")
                return

            # Calculate split sizes
            total_size = len(full_dataset)
            val_ratio = args.val_split_ratio  # Default is 0.2 (20% for validation)
            val_size = int(val_ratio * total_size)
            train_size = total_size - val_size

            # Create random splits with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(args.random_seed)
            train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

            logging.info(f"Split dataset: {train_size} samples for training, {val_size} samples for validation")

            if args.eval_interval <= 0:
                logging.info("Evaluation interval is 0. Skipping evaluation setup.")
                eval_dataset = None

    except Exception as e:
        logging.exception(f"Error initializing dataset from {args.data_dir}")
        return

    # --- Create Training DataLoader ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if device.type in ['cuda', 'mps'] else False,
        collate_fn=custom_collate_fn
    )
    logging.info(f"Training dataloader created with {len(train_dataset)} samples.")

    # --- Create Evaluation DataLoader ---
    eval_dataloader = None
    if args.eval_interval > 0 and eval_dataset is not None and len(eval_dataset) > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if device.type in ['cuda', 'mps'] else False,
            collate_fn=custom_collate_fn
        )
        logging.info(f"Evaluation dataloader created with {len(eval_dataset)} samples.")


    # --- Model Initialization ---
    model = DiffusionPolicyModel(
        state_dim=args.state_dim, time_emb_dim=args.time_emb_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_mlp_layers, image_feature_dim=args.image_feature_dim,
        use_pretrained_resnet=args.use_pretrained_resnet, freeze_resnet=args.freeze_resnet
    ).to(device)
    logging.info("Model initialized.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {num_params:,}")

    # --- State Augmentation Setup ---
    clip_bounds = None
    if args.state_aug_clip_min is not None and args.state_aug_clip_max is not None:
        clip_bounds = (args.state_aug_clip_min, args.state_aug_clip_max)

    state_augmenter = StateAugmenter(
        enabled=args.state_aug_enabled,
        noise_type=args.state_aug_noise_type,
        noise_scale=args.state_aug_noise_scale,
        noise_schedule=args.state_aug_noise_schedule,
        clip_bounds=clip_bounds,
        random_drop_prob=args.state_aug_random_drop_prob
    )

    if args.state_aug_enabled:
        logging.info(f"State augmentation enabled: {args.state_aug_noise_type} noise with scale {args.state_aug_noise_scale}")
        if args.state_aug_noise_schedule != 'constant':
            logging.info(f"Using {args.state_aug_noise_schedule} noise schedule")
        if args.state_aug_random_drop_prob > 0:
            logging.info(f"Random state dropout probability: {args.state_aug_random_drop_prob}")
        if clip_bounds:
            logging.info(f"State clipping bounds: {clip_bounds}")
    else:
        logging.info("State augmentation disabled")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logging.info(f"Optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})")

    # --- Loss Function (for training) ---
    train_criterion = nn.MSELoss()
    logging.info("Training loss function: MSELoss (on noise)")

    # --- Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Model checkpoints will be saved to: {args.output_dir}")

    # --- Resume from checkpoint if specified ---
    start_epoch = 0
    global_step = 0
    best_eval_metric = float('inf') # Now tracking MSE

    # --- Early Stopping Setup ---
    early_stopping_counter = 0
    early_stopping_best_metric = float('inf')
    early_stopping_best_model_state = None

    if args.early_stopping:
        logging.info(f"Early stopping enabled with patience={args.patience}, min_delta={args.min_delta}")
        if args.restore_best_weights:
            logging.info("Will restore model to best weights when early stopping occurs")
        if args.eval_interval <= 0:
            logging.warning("Early stopping requires evaluation. Setting eval_interval to 5.")
            args.eval_interval = 5

    if hasattr(args, 'resume_from') and args.resume_from:
        if os.path.isfile(args.resume_from):
            logging.info(f"Loading checkpoint from {args.resume_from}")
            try:
                # Load checkpoint without weights_only parameter for compatibility with older PyTorch versions
                checkpoint = torch.load(args.resume_from, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                if 'eval_metric' in checkpoint:
                    best_eval_metric = checkpoint['eval_metric']
                    logging.info(f"Resuming from epoch {start_epoch} with best eval metric: {best_eval_metric:.4f}")
                else:
                    logging.info(f"Resuming from epoch {start_epoch}")

                # Load state augmentation parameters if they exist in the checkpoint
                if args.state_aug_enabled and 'state_aug_params' in checkpoint:
                    aug_params = checkpoint['state_aug_params']
                    logging.info("Loading state augmentation parameters from checkpoint")

                    # Update state augmenter with saved parameters
                    state_augmenter.enabled = aug_params.get('enabled', state_augmenter.enabled)
                    state_augmenter.noise_type = aug_params.get('noise_type', state_augmenter.noise_type)
                    state_augmenter.noise_scale = aug_params.get('noise_scale', state_augmenter.noise_scale)
                    state_augmenter.base_noise_scale = aug_params.get('base_noise_scale', state_augmenter.base_noise_scale)
                    state_augmenter.noise_schedule = aug_params.get('noise_schedule', state_augmenter.noise_schedule)
                    state_augmenter.clip_bounds = aug_params.get('clip_bounds', state_augmenter.clip_bounds)
                    state_augmenter.random_drop_prob = aug_params.get('random_drop_prob', state_augmenter.random_drop_prob)

                    logging.info(f"Restored state augmentation: {state_augmenter.noise_type} noise with scale {state_augmenter.noise_scale}")

                # Load early stopping state if it exists in the checkpoint
                if args.early_stopping and 'early_stopping' in checkpoint:
                    es_params = checkpoint['early_stopping']
                    logging.info("Loading early stopping state from checkpoint")

                    early_stopping_counter = es_params.get('counter', early_stopping_counter)
                    early_stopping_best_metric = es_params.get('best_metric', early_stopping_best_metric)

                    if 'best_model_state' in es_params and args.restore_best_weights:
                        early_stopping_best_model_state = es_params['best_model_state']
                        logging.info("Restored early stopping best model state")

                    logging.info(f"Restored early stopping state: counter={early_stopping_counter}, best_metric={early_stopping_best_metric:.4f}")

                # Adjust learning rate if needed
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate

            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                logging.warning("Starting training from scratch")
        else:
            logging.warning(f"Checkpoint file {args.resume_from} not found. Starting training from scratch.")

    # --- Training Loop ---
    logging.info("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        batches_processed_this_epoch = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=True)

        # Update state augmentation schedule if enabled
        if args.state_aug_enabled and args.state_aug_noise_schedule != 'constant':
            state_augmenter.set_training_progress(epoch, args.num_epochs)
            if epoch % 10 == 0:  # Log every 10 epochs
                logging.info(f"State augmentation noise scale updated to {state_augmenter.noise_scale:.5f}")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            if batch is None or batch == (None, None): continue
            try: state_batch, image_batch = batch
            except Exception as e: logging.error(f"Error unpacking training batch {batch_idx}: {e}"); continue
            if state_batch is None or image_batch is None: continue

            try:
                state_batch = state_batch.to(device)
                image_batch = image_batch.to(device)
            except Exception as e: logging.error(f"Error moving training batch {batch_idx} to device {device}: {e}"); continue
            if state_batch.shape[0] == 0 or image_batch.shape[0] == 0: continue

            # --- Apply State Augmentation (if enabled) ---
            if args.state_aug_enabled:
                # Apply augmentation to the clean state before adding diffusion noise
                state_batch = state_augmenter(state_batch)

            # --- Training Step (predict noise) ---
            current_batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            noisy_state_batch = q_sample(
                x_start=state_batch, t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            predicted_noise = model(
                state=noisy_state_batch, timestep=t, image_input=image_batch
            )
            loss = train_criterion(predicted_noise, noise) # Use training criterion (MSE on noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            batches_processed_this_epoch += 1
            progress_bar.set_postfix(loss=loss.item())


        # --- End of Epoch ---
        if batches_processed_this_epoch > 0:
             avg_epoch_loss = epoch_loss / batches_processed_this_epoch
             logging.info(f"Epoch {epoch+1}/{args.num_epochs} Training Avg Loss (Noise MSE): {avg_epoch_loss:.4f}")
        else:
             logging.warning(f"Epoch {epoch+1}/{args.num_epochs} completed without processing any training batches.")
             avg_epoch_loss = float('inf')

        # --- Clean MPS Cache after each epoch ---
        if clean_mps_cache():
            logging.info("MPS cache cleaned after epoch completion")

        # --- Evaluation Step (Sampling) ---
        if eval_dataloader is not None and (epoch + 1) % args.eval_interval == 0:
            logging.info(f"--- Starting evaluation sampling for Epoch {epoch+1} ({args.num_eval_samples} samples) ---")
            avg_eval_mse = evaluate( # Now returns MSE
                model, eval_dataloader, device,
                timesteps, betas, sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas, posterior_variance, args.num_eval_samples
            )
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} Evaluation Avg State MSE: {avg_eval_mse:.4f}")

            # Save best model based on eval MSE (lower is better)
            if avg_eval_mse < best_eval_metric:
                best_eval_metric = avg_eval_mse
                best_checkpoint_path = os.path.join(args.output_dir, "model_best.pth")
                try:
                    # Save additional state augmentation info if enabled
                    checkpoint_data = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'eval_metric': avg_eval_mse, # Store eval MSE
                        'args': vars(args)
                    }

                    if args.state_aug_enabled:
                        checkpoint_data['state_aug_params'] = {
                            'enabled': state_augmenter.enabled,
                            'noise_type': state_augmenter.noise_type,
                            'noise_scale': state_augmenter.noise_scale,
                            'base_noise_scale': state_augmenter.base_noise_scale,
                            'noise_schedule': state_augmenter.noise_schedule,
                            'clip_bounds': state_augmenter.clip_bounds,
                            'random_drop_prob': state_augmenter.random_drop_prob
                        }

                    # Save early stopping state if enabled
                    if args.early_stopping:
                        es_state = {
                            'counter': early_stopping_counter,
                            'best_metric': early_stopping_best_metric
                        }

                        # Only save best model state if restore_best_weights is enabled
                        if args.restore_best_weights and early_stopping_best_model_state is not None:
                            es_state['best_model_state'] = early_stopping_best_model_state

                        checkpoint_data['early_stopping'] = es_state

                    # Save checkpoint without weights_only parameter for compatibility with older PyTorch versions
                    torch.save(checkpoint_data, best_checkpoint_path)
                    logging.info(f"Saved new best model checkpoint to {best_checkpoint_path} (Eval State MSE: {best_eval_metric:.4f})")
                except Exception as e:
                     logging.error(f"Failed to save best checkpoint at epoch {epoch+1}: {e}")

            # --- Early Stopping Logic ---
            if args.early_stopping:
                # Check if there's an improvement
                # Improvement happens when the metric decreases by at least min_delta
                if early_stopping_best_metric - avg_eval_mse > args.min_delta:
                    early_stopping_best_metric = avg_eval_mse
                    early_stopping_counter = 0

                    # Store the best model weights if restore_best_weights is enabled
                    if args.restore_best_weights:
                        early_stopping_best_model_state = {
                            key: value.cpu().clone() for key, value in model.state_dict().items()
                        }

                    logging.info(f"Early stopping: Improvement detected, counter reset (best metric: {early_stopping_best_metric:.4f})")
                else:
                    early_stopping_counter += 1
                    logging.info(f"Early stopping: No improvement, counter increased to {early_stopping_counter}/{args.patience}")

                    # Check if we should stop training
                    if early_stopping_counter >= args.patience:
                        logging.info(f"Early stopping triggered after {args.patience} evaluations without improvement")

                        # Restore best weights if enabled
                        if args.restore_best_weights and early_stopping_best_model_state is not None:
                            logging.info("Restoring model to best weights")
                            model.load_state_dict(early_stopping_best_model_state)

                        # Break out of the training loop
                        break

            logging.info(f"--- Finished evaluation sampling for Epoch {epoch+1} ---")

            # Clean MPS cache after evaluation
            if clean_mps_cache():
                logging.info("MPS cache cleaned after evaluation")


        # --- Save Regular Checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            try:
                # Save additional state augmentation info if enabled
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_epoch_loss, # Store training loss
                    'args': vars(args)
                }

                if args.state_aug_enabled:
                    checkpoint_data['state_aug_params'] = {
                        'enabled': state_augmenter.enabled,
                        'noise_type': state_augmenter.noise_type,
                        'noise_scale': state_augmenter.noise_scale,
                        'base_noise_scale': state_augmenter.base_noise_scale,
                        'noise_schedule': state_augmenter.noise_schedule,
                        'clip_bounds': state_augmenter.clip_bounds,
                        'random_drop_prob': state_augmenter.random_drop_prob
                    }

                # Save early stopping state if enabled
                if args.early_stopping:
                    es_state = {
                        'counter': early_stopping_counter,
                        'best_metric': early_stopping_best_metric
                    }

                    # Only save best model state if restore_best_weights is enabled
                    if args.restore_best_weights and early_stopping_best_model_state is not None:
                        es_state['best_model_state'] = early_stopping_best_model_state

                    checkpoint_data['early_stopping'] = es_state

                # Save checkpoint without weights_only parameter for compatibility with older PyTorch versions
                torch.save(checkpoint_data, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                 logging.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}")


    # Final training message
    if args.early_stopping and early_stopping_counter >= args.patience:
        logging.info(f"Training finished early due to early stopping after {epoch+1} epochs.")
        logging.info(f"Best validation metric: {early_stopping_best_metric:.4f}")
    else:
        logging.info(f"Training finished after completing all {args.num_epochs} epochs.")
        logging.info(f"Best validation metric: {best_eval_metric:.4f}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Policy Model")

    # Paths and Directories
    parser.add_argument('--data_dir', type=str, default='/Users/lambertwang/Downloads/FedVLA_latest/mycobot_episodes', help='Base directory for training dataset')
    parser.add_argument('--eval_data_dir', type=str, default='/Users/lambertwang/Downloads/FedVLA_latest/mycobot_episodes', help='Base directory for evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--num_episodes', type=int, default=300, help='Number of episodes to load for training')
    parser.add_argument('--eval_num_episodes', type=int, default=32, help='Number of episodes for evaluation (uses num_episodes if None)')

    # Model Hyperparameters
    parser.add_argument('--state_dim', type=int, default=7, help='Dimension of the state vector')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to')
    parser.add_argument('--image_feature_dim', type=int, default=512, help='Feature dimension from ResNet backbone')
    parser.add_argument('--time_emb_dim', type=int, default=64, help='Dimension for timestep embedding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for MLP layers')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help='Number of MLP layers')
    parser.add_argument('--use_pretrained_resnet', action='store_true', help='Use pretrained ResNet weights')
    parser.add_argument('--no_use_pretrained_resnet', action='store_false', dest='use_pretrained_resnet', help='Do not use pretrained ResNet weights')
    parser.add_argument('--freeze_resnet', action='store_true', help='Freeze ResNet backbone weights')
    parser.add_argument('--no_freeze_resnet', action='store_false', dest='freeze_resnet', help='Do not freeze ResNet backbone weights')
    parser.set_defaults(use_pretrained_resnet=True, freeze_resnet=True)

    # Diffusion Hyperparameters
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Total number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Starting value for linear beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Ending value for linear beta schedule')

    # Training Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation sampling (often needs to be smaller due to sampling loop memory)') # Reduced eval batch size
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--save_interval', type=int, default=20, help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate model every N epochs (set to 0 to disable)')
    parser.add_argument('--num_eval_samples', type=int, default=32, help='Number of samples to generate during evaluation') # Added num_eval_samples
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use if available')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint file to resume training from')

    # State Augmentation Parameters
    parser.add_argument('--state_aug_enabled', action='store_true', help='Enable state augmentation during training')
    parser.add_argument('--state_aug_noise_type', type=str, default='gaussian', choices=['gaussian', 'uniform', 'scaled'], help='Type of noise to apply for state augmentation')
    parser.add_argument('--state_aug_noise_scale', type=float, default=0.05, help='Scale of noise to apply for state augmentation')
    parser.add_argument('--state_aug_noise_schedule', type=str, default='cosine_decay', choices=['constant', 'linear_decay', 'cosine_decay'], help='How noise scale changes over training')
    parser.add_argument('--state_aug_random_drop_prob', type=float, default=0.0, help='Probability of randomly zeroing out a joint value (0.0 to disable)')
    parser.add_argument('--state_aug_clip_min', type=float, default=None, help='Minimum value to clip augmented state (None for no clipping)')
    parser.add_argument('--state_aug_clip_max', type=float, default=None, help='Maximum value to clip augmented state (None for no clipping)')

    # Early Stopping Parameters
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping based on validation performance')
    parser.add_argument('--patience', type=int, default=15, help='Number of evaluations to wait for improvement before stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in validation metric to qualify as improvement')
    parser.add_argument('--restore_best_weights', action='store_true', help='Restore model to best weights when early stopping occurs')

    # Add this to your argument parser
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory for faster data transfer to GPU')
    parser.set_defaults(pin_memory=True)  # Enable by default

    # Dataset splitting parameters
    parser.add_argument('--val_split_ratio', type=float, default=0.2, help='Ratio of data to use for validation when using a single dataset (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible dataset splitting (default: 42)')

    args = parser.parse_args()

    if os.name == 'nt' and args.num_workers > 0:
         logging.warning("Setting num_workers > 0 on Windows can cause issues. Forcing num_workers = 0.")
         args.num_workers = 0

    # Ensure eval_interval > 0 if we want evaluation
    if args.eval_interval <= 0:
         logging.info("Evaluation interval is <= 0. Evaluation will be disabled.")
         args.eval_dataloader = None # Explicitly disable

    logging.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")

    train(args)
