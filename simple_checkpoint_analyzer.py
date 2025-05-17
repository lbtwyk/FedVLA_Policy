#!/usr/bin/env python3
"""
Simple Diffusion Policy Checkpoint Analyzer

A lightweight script to analyze Diffusion Policy model checkpoints without external dependencies.

Usage:
    python simple_checkpoint_analyzer.py --checkpoint_dir ./checkpoints
    python simple_checkpoint_analyzer.py --file ./checkpoints/model_best.pth
    python simple_checkpoint_analyzer.py --compare --checkpoint_dir ./checkpoints
"""

import os
import sys
import glob
import re
import argparse
import torch
from typing import Dict, List, Any, Optional, Union, Tuple

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint file and return its contents.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    try:
        # First try to add safe globals for PyTorch 2.6+ compatibility
        try:
            import torch.serialization
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
        except (ImportError, AttributeError):
            pass

        # Try different loading methods
        try:
            # Method 1: Standard loading
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e1:
            try:
                # Method 2: Explicitly set weights_only=False for PyTorch 2.6+
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e2:
                try:
                    # Method 3: Use pickle directly as a last resort
                    import pickle
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                except Exception as e3:
                    print(f"Failed to load checkpoint using all methods:")
                    print(f"  Method 1: {e1}")
                    print(f"  Method 2: {e2}")
                    print(f"  Method 3: {e3}")
                    return {}

        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return {}

def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch number from checkpoint filename."""
    match = re.search(r'model_epoch_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    return None

def format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        return f"{value:.6f}"
    elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], float):
        return f"[{', '.join([f'{v:.4f}' for v in value])}]"
    elif isinstance(value, (dict)):
        return str({k: format_value(v) for k, v in value.items()})
    else:
        return str(value)

def analyze_single_checkpoint(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a single checkpoint file and display its information.

    Args:
        checkpoint_path: Path to the checkpoint file
        verbose: Whether to print detailed information

    Returns:
        Dictionary with extracted information
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return {}

    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint:
        return {}

    # Extract basic information
    result = {
        'file': os.path.basename(checkpoint_path),
        'epoch': checkpoint.get('epoch', None),
        'eval_metric': checkpoint.get('eval_metric', None),
        'train_loss': checkpoint.get('train_loss', None),
    }

    # If epoch not in checkpoint data, try to extract from filename
    if result['epoch'] is None:
        result['epoch'] = extract_epoch_from_filename(os.path.basename(checkpoint_path))

    # Extract state augmentation parameters if available
    if 'state_aug_params' in checkpoint:
        result['state_aug_params'] = checkpoint['state_aug_params']

    # Extract early stopping information if available
    if 'early_stopping' in checkpoint:
        result['early_stopping'] = checkpoint['early_stopping']

    # Extract training arguments if available
    if 'args' in checkpoint:
        result['args'] = checkpoint['args']

    if verbose:
        print(f"\n{'='*80}")
        print(f"CHECKPOINT ANALYSIS: {os.path.basename(checkpoint_path)}")
        print(f"{'='*80}")

        # Basic information
        print(f"\n--- Basic Information ---")
        print(f"Epoch: {result['epoch']}")
        if result['eval_metric'] is not None:
            print(f"Validation MSE: {result['eval_metric']:.6f}")
        if result['train_loss'] is not None:
            print(f"Training Loss: {result['train_loss']:.6f}")

        # State augmentation parameters
        if 'state_aug_params' in result:
            print(f"\n--- State Augmentation Parameters ---")
            for key, value in result['state_aug_params'].items():
                print(f"{key}: {format_value(value)}")

        # Early stopping information
        if 'early_stopping' in result:
            print(f"\n--- Early Stopping Information ---")
            es_info = result['early_stopping']
            for key, value in es_info.items():
                if key != 'best_model_state':  # Skip the model state
                    print(f"{key}: {format_value(value)}")

        # Training arguments (selected important ones)
        if 'args' in result:
            print(f"\n--- Training Configuration ---")
            important_args = [
                'learning_rate', 'batch_size', 'num_epochs', 'weight_decay',
                'diffusion_timesteps', 'beta_start', 'beta_end',
                'state_aug_enabled', 'state_aug_noise_type', 'state_aug_noise_scale',
                'early_stopping', 'patience', 'min_delta'
            ]

            for arg in important_args:
                if arg in result['args']:
                    print(f"{arg}: {format_value(result['args'][arg])}")

            # Print data directory
            if 'data_dir' in result['args']:
                print(f"data_dir: {result['args']['data_dir']}")

    return result

def compare_checkpoints(checkpoint_dir: str) -> None:
    """
    Compare metrics across multiple checkpoint files.

    Args:
        checkpoint_dir: Directory containing checkpoint files
    """
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pth'))
    best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        checkpoint_files.append(best_model_path)

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Analyze each checkpoint
    results = []
    for checkpoint_path in checkpoint_files:
        result = analyze_single_checkpoint(checkpoint_path, verbose=False)
        if result:
            results.append(result)

    # Sort by epoch
    results.sort(key=lambda x: x['epoch'] if x['epoch'] is not None else float('inf'))

    # Print header
    print("\n--- Checkpoint Comparison ---")
    header_format = "{:<25} {:<10} {:<20} {:<20}"
    print(header_format.format("File", "Epoch", "Validation MSE", "Training Loss"))
    print("-" * 75)

    # Print data
    row_format = "{:<25} {:<10} {:<20} {:<20}"
    for result in results:
        epoch = result['epoch'] if result['epoch'] is not None else 'Unknown'
        eval_metric = f"{result['eval_metric']:.6f}" if result['eval_metric'] is not None else 'N/A'
        train_loss = f"{result['train_loss']:.6f}" if result['train_loss'] is not None else 'N/A'

        print(row_format.format(result['file'], epoch, eval_metric, train_loss))

    # Find best validation metric
    eval_results = [(r['epoch'], r['eval_metric']) for r in results if r['eval_metric'] is not None]
    if eval_results:
        best_epoch, best_metric = min(eval_results, key=lambda x: x[1])
        best_file = next(r['file'] for r in results if r['epoch'] == best_epoch)
        print(f"\nBest validation performance: Epoch {best_epoch} with MSE: {best_metric:.6f} (file: {best_file})")

def main():
    parser = argparse.ArgumentParser(description='Analyze Diffusion Policy model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific checkpoint file to analyze')
    parser.add_argument('--compare', action='store_true',
                        help='Compare metrics across multiple checkpoints')

    args = parser.parse_args()

    # Analyze a specific file
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        analyze_single_checkpoint(args.file)

    # Compare multiple checkpoints
    elif args.compare:
        compare_checkpoints(args.checkpoint_dir)

    # Default: analyze best model
    else:
        best_model_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
        if os.path.exists(best_model_path):
            analyze_single_checkpoint(best_model_path)
        else:
            print(f"Best model checkpoint not found at {best_model_path}")
            print("Use --compare to analyze all available checkpoints")

if __name__ == "__main__":
    main()
