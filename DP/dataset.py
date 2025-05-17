# FedVLA/DP/dataset.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Any, Optional, Callable # Added Callable
import logging
from PIL import Image # Import Pillow library

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RobotEpisodeDataset(Dataset):
    """
    A PyTorch Dataset to load robot trajectory data from multiple episodes.

    Each episode consists of state information (joint angles, gripper state)
    and corresponding image frames stored in a specific directory structure.
    This version loads and transforms images.
    """
    def __init__(self,
                 base_dir: str,
                 num_episodes: int,
                 episode_prefix: str = 'episode',
                 transform: Optional[Callable] = None, # Accept an optional transform
                 flat_structure: bool = False # Flag to indicate if the dataset has a flat structure
                ):
        """
        Initializes the dataset.

        Args:
            base_dir (str): Base directory containing episode subdirectories.
            num_episodes (int): Total number of episodes to load.
            episode_prefix (str, optional): Prefix for episode directory names. Defaults to 'episode'.
            transform (callable, optional): Optional transform to be applied
                                             on a sample image. Defaults to None.
            flat_structure (bool, optional): If True, assumes episodes are directly in base_dir.
                                            If False (default), assumes collection/episode nested structure.
        """
        super().__init__()
        self.base_dir = base_dir
        self.num_episodes = num_episodes
        self.episode_prefix = episode_prefix
        self.transform = transform # Store the transform
        self.flat_structure = flat_structure # Store the structure type

        self.data_points: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        """
        Scans for episodes, parses states.json files, and aggregates all timesteps into the self.data_points list.
        Handles both nested (collection/episode) and flat directory structures.
        Stores image paths for later loading.
        """
        structure_type = "flat" if self.flat_structure else "nested collection/episode"
        logging.info(f"Scanning data from base directory: {self.base_dir} using {structure_type} structure.")
        total_timesteps = 0
        skipped_timesteps = 0
        loaded_episodes_count = 0

        if self.flat_structure:
            # Flat structure: episodes are directly in the base directory
            self._load_flat_structure(loaded_episodes_count, total_timesteps, skipped_timesteps)
        else:
            # Nested structure: collection folders contain episode folders
            self._load_nested_structure(loaded_episodes_count, total_timesteps, skipped_timesteps)

        logging.info(f"Finished scanning data. Processed {loaded_episodes_count} episode structures. Found {len(self.data_points)} valid timestep entries.")
        if skipped_timesteps > 0:
            logging.warning(f"Skipped {skipped_timesteps} timesteps due to missing/invalid data or files.")
        if not self.data_points:
             logging.warning("No data points were loaded. Check base directory and episode structure.")

    def _load_nested_structure(self, loaded_episodes_count, total_timesteps, skipped_timesteps):
        """
        Loads data from a nested directory structure (collection/episode).
        """
        try:
            collection_dirs_all = sorted([
                d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('collection_')
            ])
        except FileNotFoundError:
            logging.error(f"Base directory not found: {self.base_dir}")
            return
        except Exception as e:
            logging.error(f"Error listing base directory {self.base_dir}: {e}")
            return

        # Limit the number of collection directories to process based on self.num_episodes
        collection_dirs_to_process = collection_dirs_all[:self.num_episodes if self.num_episodes > 0 else len(collection_dirs_all)]

        for collection_name in collection_dirs_to_process:
            collection_path = os.path.join(self.base_dir, collection_name)

            try:
                potential_episode_dirs = [
                    d for d in os.listdir(collection_path)
                    if os.path.isdir(os.path.join(collection_path, d)) and d.startswith(self.episode_prefix)
                ]
            except Exception as e:
                logging.warning(f"Could not list contents of {collection_path}: {e}, skipping collection.")
                continue

            if not potential_episode_dirs:
                logging.warning(f"No directory starting with '{self.episode_prefix}' found in {collection_path}, skipping collection.")
                continue

            if len(potential_episode_dirs) > 1:
                logging.warning(f"Multiple '{self.episode_prefix}' directories found in {collection_path}. Using the first one: {potential_episode_dirs[0]}.")

            episode_name = potential_episode_dirs[0] # Use the first one found
            episode_dir = os.path.join(collection_path, episode_name) # This is the actual directory with states.json

            # Process the episode directory
            episode_result = self._process_episode_dir(episode_dir, loaded_episodes_count)
            if episode_result:
                loaded_episodes_count += 1
                total_timesteps += episode_result['timesteps']
                skipped_timesteps += episode_result['skipped']

    def _load_flat_structure(self, loaded_episodes_count, total_timesteps, skipped_timesteps):
        """
        Loads data from a flat directory structure (episodes directly in base_dir).
        """
        try:
            episode_dirs_all = sorted([
                d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith(self.episode_prefix)
            ])
        except FileNotFoundError:
            logging.error(f"Base directory not found: {self.base_dir}")
            return
        except Exception as e:
            logging.error(f"Error listing base directory {self.base_dir}: {e}")
            return

        # Limit the number of episode directories to process
        episode_dirs_to_process = episode_dirs_all[:self.num_episodes if self.num_episodes > 0 else len(episode_dirs_all)]

        for episode_name in episode_dirs_to_process:
            episode_dir = os.path.join(self.base_dir, episode_name)

            # Process the episode directory
            episode_result = self._process_episode_dir(episode_dir, loaded_episodes_count)
            if episode_result:
                loaded_episodes_count += 1
                total_timesteps += episode_result['timesteps']
                skipped_timesteps += episode_result['skipped']

    def _process_episode_dir(self, episode_dir, episode_index):
        """
        Processes a single episode directory, extracting state and image data.

        Args:
            episode_dir (str): Path to the episode directory
            episode_index (int): Current episode count for indexing

        Returns:
            dict: Contains 'timesteps' and 'skipped' counts, or None if processing failed
        """
        state_file_path = os.path.join(episode_dir, 'states.json')

        if not os.path.isfile(state_file_path):
            logging.warning(f"states.json not found in {episode_dir}, skipping this episode.")
            return None

        episode_timestep_count = 0
        skipped_timesteps = 0

        try:
            with open(state_file_path, 'r') as f:
                episode_data = json.load(f)

            if not isinstance(episode_data, list):
                logging.warning(f"states.json in {episode_dir} is not a list, skipping episode data.")
                return None

            for timestep_data in episode_data:
                if not all(k in timestep_data for k in ["angles", "gripper_value", "image"]):
                    logging.warning(f"Missing keys in timestep data in {state_file_path}, skipping timestep.")
                    skipped_timesteps += 1
                    continue
                if not isinstance(timestep_data["angles"], list) or len(timestep_data["angles"]) != 6:
                     logging.warning(f"Invalid 'angles' format in {state_file_path}, skipping timestep.")
                     skipped_timesteps += 1
                     continue
                if not isinstance(timestep_data["gripper_value"], list) or len(timestep_data["gripper_value"]) != 1:
                     logging.warning(f"Invalid 'gripper_value' format in {state_file_path}, skipping timestep.")
                     skipped_timesteps += 1
                     continue
                if not isinstance(timestep_data["image"], str):
                     logging.warning(f"Invalid 'image' format (not a string) in {state_file_path}, skipping timestep.")
                     skipped_timesteps += 1
                     continue

                relative_image_path = timestep_data["image"]
                # Image path is relative to episode_dir (e.g., "frame_dir/image_0000.png")
                full_image_path = os.path.join(episode_dir, relative_image_path)

                if not os.path.isfile(full_image_path):
                    logging.warning(f"Image file not found during scan: {full_image_path}, skipping timestep.")
                    skipped_timesteps += 1
                    continue

                self.data_points.append({
                    'angles': timestep_data['angles'],
                    'gripper_value': timestep_data['gripper_value'][0],
                    'image_path': full_image_path,
                    'episode_index': episode_index + 1, # Use the count of processed episodes
                })
                episode_timestep_count += 1

            return {
                'timesteps': episode_timestep_count,
                'skipped': skipped_timesteps
            }

        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {state_file_path}, skipping episode data.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred processing {episode_dir}: {e}, skipping episode data.")
            return None


    def __len__(self) -> int:
        """
        Returns the total number of valid timesteps found during the scan.
        """
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves the data for a single timestep, including the loaded and transformed image.

        Args:
            idx (int): The index of the timestep to retrieve.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: A tuple containing:
                - state_vector (torch.Tensor): State tensor (7 dimensions, float32).
                - image_tensor (torch.Tensor): Transformed image tensor (e.g., [3, H, W]).
            Returns None if there's an error loading/processing the image for this index.
        """
        if idx < 0 or idx >= len(self.data_points):
            logging.error(f"Index {idx} out of bounds for dataset with length {len(self)}")
            # Returning None might be handled by the collate_fn, but raising is clearer
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")

        data_point = self.data_points[idx]

        # --- State Vector ---
        try:
            # Combine angles and gripper value into a single state vector
            state_vector_list = data_point['angles'] + [float(data_point['gripper_value'])]
            state_vector = torch.tensor(state_vector_list, dtype=torch.float32) # Shape (7,)
        except Exception as e:
             logging.error(f"Error processing state for index {idx}, data: {data_point}. Error: {e}")
             return None # Signal error for this item

        # --- Image Loading and Transformation ---
        image_path = data_point['image_path']
        try:
            # Load image using Pillow
            image = Image.open(image_path).convert('RGB') # Ensure image is RGB

            # Apply transformations if provided
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # If no transform provided, convert to tensor manually (less common)
                # Note: This won't have normalization or resizing needed for ResNet
                image_tensor = transforms.ToTensor()(image)

            # Basic check on tensor shape (optional but good practice)
            if not isinstance(image_tensor, torch.Tensor) or image_tensor.dim() != 3:
                 logging.warning(f"Image transform for index {idx} (path: {image_path}) did not produce a 3D tensor. Got shape {image_tensor.shape if isinstance(image_tensor, torch.Tensor) else type(image_tensor)}. Skipping item.")
                 return None # Signal error

        except FileNotFoundError:
            logging.error(f"Image file not found for index {idx} at path: {image_path}. This shouldn't happen if scan was accurate.")
            return None # Signal error for this item
        except Exception as e:
            logging.error(f"Error loading or transforming image for index {idx} (path: {image_path}): {e}")
            return None # Signal error for this item

        return state_vector, image_tensor

# --- Example Usage (Optional - for testing dataset directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for more detailed dataset logs

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_dir = os.path.join(script_dir, '..', 'stack_orange')
    base_data_directory = default_base_dir
    num_episodes_to_load = 5 # Load fewer episodes for faster testing

    print(f"Attempting to load data from: {base_data_directory}")
    print("-" * 30)

    # Define the same transform used in train.py for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.isdir(base_data_directory):
        print(f"ERROR: Base directory '{base_data_directory}' not found.")
    else:
        # Instantiate with the transform
        robot_dataset = RobotEpisodeDataset(
            base_dir=base_data_directory,
            num_episodes=num_episodes_to_load,
            transform=test_transform
        )

        print("-" * 30)
        print(f"Dataset initialized.")

        if len(robot_dataset) == 0:
            print("Dataset is empty. Please check logs and data structure.")
        else:
            print(f"Total number of loadable timesteps found: {len(robot_dataset)}")

            # Test getting a few samples
            print("\n--- Sample Data Test ---")
            indices_to_check = [0, len(robot_dataset) // 2, len(robot_dataset) - 1]
            valid_samples = 0
            for i in indices_to_check:
                 if i >= len(robot_dataset): continue # Skip if index out of bounds
                 print(f"\nAttempting to get Sample Index: {i}")
                 try:
                     # __getitem__ now returns Optional[...]
                     sample = robot_dataset[i]
                     if sample is not None:
                         state, image = sample
                         print(f"  Successfully retrieved sample {i}:")
                         print(f"    State Tensor Shape: {state.shape}, Dtype: {state.dtype}")
                         print(f"    Image Tensor Shape: {image.shape}, Dtype: {image.dtype}")
                         # Check image value range (after ToTensor, before Normalize: [0,1])
                         # After Normalize, range will vary.
                         print(f"    Image Tensor Min/Max: {image.min():.2f}/{image.max():.2f}")
                         valid_samples += 1
                     else:
                         print(f"  Failed to retrieve sample {i} (returned None). Check logs for errors.")
                 except IndexError as e:
                     print(f"  Error getting sample {i}: {e}")
                 except Exception as e:
                     print(f"  Unexpected error getting sample {i}: {e}")
                     logging.exception("Error during __getitem__ test")


            print(f"\nSuccessfully retrieved {valid_samples}/{len(indices_to_check)} tested samples.")
            print("--- End Sample Data Test ---")

            # Optional: Test with DataLoader
            print("\n--- DataLoader Test ---")
            try:
                # Define a simple collate function for testing
                def test_collate_fn(batch):
                    """Simple collate function for testing."""
                    states = []
                    images = []
                    for item in batch:
                        if item is None or not isinstance(item, tuple) or len(item) != 2:
                            continue
                        state, image = item
                        states.append(state)
                        images.append(image)
                    if not states:
                        return None, None
                    return torch.stack(states), torch.stack(images)

                # Use our test collate function
                test_loader = DataLoader(robot_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=test_collate_fn) # num_workers=0 for easier debugging
                batch_count = 0
                for batch in test_loader:
                    if batch is None or batch == (None, None):
                        print("  DataLoader yielded an empty/invalid batch (collate_fn filtering).")
                        continue
                    state_b, image_b = batch
                    print(f"  Loaded batch {batch_count}: State shape {state_b.shape}, Image shape {image_b.shape}")
                    batch_count += 1
                    if batch_count >= 2: # Just check a couple of batches
                        break
                print("DataLoader test finished.")
            except Exception as e:
                print(f"Error during DataLoader test: {e}")
                logging.exception("DataLoader test failed")

