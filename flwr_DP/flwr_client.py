from collections import OrderedDict
import flwr as fl
from flwr.client import ClientApp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import logging
import os
import argparse
from dataset import RobotEpisodeDataset
from model import DiffusionPolicyModel
from train import linear_beta_schedule, q_sample, custom_collate_fn
from tqdm import tqdm
import datetime
import gc # Add gc for garbage collection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create checkpoints directory if it doesn't exist
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define NUM_CLIENTS globally or pass it appropriately if it varies
NUM_TOTAL_CLIENTS = 3 # Example, adjust as needed or load from config

def clean_mps_cache(force_gc=True):
    """
    Enhanced MPS cache cleaning to prevent memory explosion on Apple Silicon hardware.
    This function should be called periodically during long training runs.
    
    Args:
        force_gc: Whether to force garbage collection (default: True)
    """
    if torch.backends.mps.is_available():
        if force_gc:
            # Force garbage collection multiple times for thorough cleanup
            for _ in range(3):
                gc.collect()
        
        # Synchronize MPS operations before clearing cache
        torch.mps.synchronize()
        
        # Empty MPS cache
        torch.mps.empty_cache()
        
        # Additional synchronization after cache clear
        torch.mps.synchronize()
        
        return True
    return False

def load_model():
    """Load the diffusion policy model."""
    model = DiffusionPolicyModel(
        state_dim=7,  # 6 joint angles + 1 gripper value
        time_emb_dim=64,
        hidden_dim=256,
        num_layers=4,
        image_feature_dim=512,
        use_pretrained_resnet=True,
        freeze_resnet=True
    )
    return model

def load_data(client_id: int, num_total_clients: int): # client_id is now an int
    """Load the training and test data for a specific client."""
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_dataset = RobotEpisodeDataset(
        base_dir='../mycobot_episodes_degrees/',
        num_episodes=300,
        transform=image_transform,
        flat_structure=True # Ensure this matches your dataset structure
    )

    # Split dataset among clients
    total_size = len(full_dataset)
    client_size = total_size // num_total_clients
    start_idx = client_id * client_size
    end_idx = start_idx + client_size if client_id < num_total_clients - 1 else total_size

    # Create client-specific dataset
    client_dataset = Subset(full_dataset, range(start_idx, end_idx))
    logging.info(f"Client {client_id}: Using {len(client_dataset)} samples out of {total_size} total samples")

    # Create train and test loaders for this client
    train_size = int(0.8 * len(client_dataset))
    test_size = len(client_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(client_dataset, [train_size, test_size])

    trainloader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced for Mac stability
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=16, # As per user's latest change
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return trainloader, testloader

def train_client_model(model, trainloader, epochs=1): # Renamed from train to avoid conflict
    """Train the model for one epoch."""
    # Device setup with MPS support for Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Metal) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device for training")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device for training")
    
    model = model.to(device)
    model.train()

    # Setup diffusion parameters
    timesteps = 1000 # Should match centralized/overall diffusion setup
    betas = linear_beta_schedule(timesteps=timesteps).to(device) # Ensure on correct device
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = nn.MSELoss()

    total_loss = 0
    num_batches = 0

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(trainloader, desc=f"Training Epoch {epoch + 1}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None or batch == (None, None):
                continue

            state_batch, image_batch = batch
            state_batch = state_batch.to(device)
            image_batch = image_batch.to(device)

            optimizer.zero_grad()

            current_batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (current_batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            noisy_state_batch = q_sample(
                x_start=state_batch,
                t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            predicted_noise = model(
                state=noisy_state_batch,
                timestep=t,
                image_input=image_batch
            )
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })

            # MPS cache cleaning periodically during training
            if device.type == "mps" and batch_idx % 50 == 0 and batch_idx > 0:
                clean_mps_cache(force_gc=False)
        
        if num_batches == 0:
            logging.warning(f"Epoch {epoch + 1} completed without processing any batches.")
            avg_loss = float('inf')
        else:
            avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}")

        # MPS cache cleaning after each epoch
        if device.type == "mps":
            clean_mps_cache(force_gc=True)
            logging.info(f"MPS cache cleaned after epoch {epoch + 1} in train_client_model for client.")
    
    return avg_loss

def test_client_model(model, testloader): # Renamed from test
    """Evaluate the model."""
    # Device setup with MPS support for Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Metal) for evaluation")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device for evaluation")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device for evaluation")

    model = model.to(device)
    model.eval()

    timesteps = 1000 # Should match centralized/overall diffusion setup
    betas = linear_beta_schedule(timesteps=timesteps).to(device) # Ensure on correct device
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    total_loss = 0
    total_samples = 0 # Changed from 'total' to avoid confusion

    logging.info("Starting evaluation...")
    progress_bar = tqdm(testloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None or batch == (None, None):
                continue

            state_batch, image_batch = batch
            state_batch = state_batch.to(device)
            image_batch = image_batch.to(device)

            batch_size = state_batch.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(state_batch)
            
            noisy_state_batch = q_sample(
                x_start=state_batch,
                t=t,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                noise=noise
            )
            
            predicted_noise = model(
                state=noisy_state_batch,
                timestep=t,
                image_input=image_batch
            )
            
            loss = nn.MSELoss()(predicted_noise, noise)
            total_loss += loss.item() * batch_size # Accumulate sum of losses
            total_samples += batch_size

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_samples:.4f}' if total_samples > 0 else 'N/A'
            })

            # MPS cache cleaning periodically during evaluation (less frequent)
            if device.type == "mps" and batch_idx % 100 == 0 and batch_idx > 0: # Less frequent than training
                clean_mps_cache(force_gc=False)
    
    if total_samples == 0:
        logging.warning("Evaluation completed without processing any samples.")
        avg_loss = float('inf')
        accuracy = 0.0
    else:
        avg_loss = total_loss / total_samples
        accuracy = 1.0 - min(1.0, avg_loss) 

    # MPS cache cleaning after evaluation is complete
    if device.type == "mps":
        clean_mps_cache(force_gc=True)
        logging.info("MPS cache cleaned after test_client_model for client.")
    
    logging.info(f"Evaluation completed - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def save_checkpoint(model, round_num, train_loss, eval_loss=None, eval_accuracy=None, client_id=None, total_epochs=None):
    """Save model checkpoint with metadata."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    client_str = f"_client{client_id}" if client_id is not None else ""
    epoch_str = f"_epoch{total_epochs}" if total_epochs is not None else ""
    
    # Ensure the client-specific checkpoint directory exists
    client_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"client_{client_id}")
    os.makedirs(client_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(client_checkpoint_dir, f"model_round_{round_num}{epoch_str}_{timestamp}.pth")
    
    checkpoint = {
        'round': round_num,
        'client_id': client_id,
        'total_epochs': total_epochs, # This is round number in FL context
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, num_total_clients: int): # cid is now int
        super().__init__()
        self.current_round = 0 # Tracks rounds for this client
        self.client_id = client_id
        self.num_total_clients = num_total_clients
        self.net = load_model()
        self.trainloader, self.testloader = load_data(self.client_id, self.num_total_clients)
        self.ckpt_interval = 200
        self.eval_interval = 200
        logging.info(f"Initialized FlowerClient {self.client_id} with {len(self.trainloader.dataset)} training samples and {len(self.testloader.dataset)} test samples")

    def get_parameters(self, config=None):
        logging.info(f"Client {self.client_id}: Getting model parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        self.current_round += 1
        # In simulation, total_epochs might be better tracked by server rounds
        # Or, if each fit is one local epoch: self.total_epochs_trained_locally += 1
        logging.info(f"Client {self.client_id}: Starting local training for round {self.current_round} (Total FL rounds so far: {self.current_round})")
        set_parameters(self.net, parameters)
        # Assuming 1 epoch of local training per federated round
        train_loss = train_client_model(self.net, self.trainloader, epochs=1) 
        logging.info(f"Client {self.client_id}: Local training completed for round {self.current_round} - Loss: {train_loss:.4f}")
        
        if self.current_round % self.ckpt_interval == 0:
            save_checkpoint(
                self.net, 
                self.current_round, 
                train_loss, 
                client_id=self.client_id,
                total_epochs=self.current_round # Using round as epoch marker for FL
            )
        
        return self.get_parameters(config), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        if self.current_round == 0 or self.current_round % self.eval_interval != 0 : # Evaluate on first round too or skip if 0
             # Return dummy values if not evaluating to meet API requirements
             logging.info(f"Client {self.client_id}: Skipping evaluation for round {self.current_round}")
             return 0.0, len(self.testloader.dataset), {"accuracy": 0.0, "eval_loss": 0.0}

        logging.info(f"Client {self.client_id}: Starting local evaluation for round {self.current_round}")
        set_parameters(self.net, parameters)
        loss, accuracy = test_client_model(self.net, self.testloader)
        logging.info(f"Client {self.client_id}: Local evaluation completed for round {self.current_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.current_round % self.ckpt_interval == 0: # Also save checkpoint during eval round if interval met
            save_checkpoint(
                self.net, 
                self.current_round, 
                None, # No specific train_loss for this call point
                loss, 
                accuracy, 
                client_id=self.client_id,
                total_epochs=self.current_round # Using round as epoch marker
            )
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy), "eval_loss": float(loss)}

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Define a function that creates a FlowerClient instance
def client_fn(cid: str) -> fl.client.Client: # cid is string from run_simulation
    """Create a Flower client."""
    client_id_int = int(cid) # Convert string cid to int
    return FlowerClient(client_id=client_id_int, num_total_clients=NUM_TOTAL_CLIENTS).to_client()

# Create ClientApp instance
app = ClientApp(
    client_fn=client_fn,
)

# No main function here anymore, it will be in run_simulation.py
# Old main() function removed.
