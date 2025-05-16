# Running Diffusion Policy on macOS with MPS Acceleration

This guide provides comprehensive instructions for running the Diffusion Policy machine learning model on macOS using Apple's Metal Performance Shaders (MPS) for GPU acceleration.

## System Requirements

- Mac with Apple Silicon (M1, M2, M3 series) or AMD GPU
- macOS 12.3 or later
- Python 3.7 or later
- Xcode Command Line Tools

## Manual Setup Instructions

Follow these step-by-step instructions to set up the Diffusion Policy environment on your Mac without relying on setup scripts.

### 1. Install Xcode Command Line Tools

If you haven't already installed the Xcode Command Line Tools, run:

```bash
xcode-select --install
```

Follow the prompts to complete the installation. This may take a few minutes.

### 2. Check Python Version

Verify that you have Python 3.7 or later installed:

```bash
python3 --version
```

If Python is not installed or the version is too old, download and install the latest version from [python.org](https://www.python.org/downloads/macos/).

### 3. Create a Virtual Environment

Navigate to the Diffusion Policy directory and create a new Python virtual environment:

```bash
# Navigate to the DP directory
cd /path/to/FedVLA_latest/DP

# Create a new virtual environment
python3 -m venv dp_venv
```

### 4. Activate the Virtual Environment

Activate the newly created virtual environment:

```bash
source dp_venv/bin/activate
```

Your command prompt should change to indicate that the virtual environment is active.

### 5. Upgrade Pip

Upgrade pip to the latest version:

```bash
pip install --upgrade pip
```

### 6. Install PyTorch with MPS Support

Install PyTorch with MPS support:

```bash
pip install torch torchvision torchaudio
```

### 7. Install Required Dependencies

Install all the required dependencies for Diffusion Policy:

```bash
pip install numpy>=1.20.0 pillow>=8.0.0 tqdm>=4.60.0 tensorboard>=2.5.0 matplotlib
```

### 8. Verify MPS Support

Create a simple Python script to verify that MPS is available and working correctly:

```bash
# Create a test script
cat > test_mps_simple.py << 'EOF'
import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    # Create a device
    device = torch.device("mps")
    print("MPS device created successfully")

    # Create tensors and perform operations
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Warm-up
    for _ in range(5):
        c = torch.matmul(a, b)

    # Benchmark
    torch.mps.synchronize()
    start_time = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.mps.synchronize()
    end_time = time.time()

    print(f"Time for 10 matrix multiplications on MPS: {end_time - start_time:.4f} seconds")
    print("MPS is working correctly!")
else:
    print("MPS is not available on this system.")
    if torch.backends.mps.is_built():
        print("PyTorch is built with MPS support, but your macOS version may not support it.")
        print("Make sure you're running macOS 12.3 or later.")
    else:
        print("PyTorch is not built with MPS support.")
        print("Please reinstall PyTorch with MPS support.")
EOF

# Run the test script
python test_mps_simple.py
```

If MPS is working correctly, you should see output confirming that MPS is available and the benchmark results.

### 9. Modify Code to Support MPS

The Diffusion Policy code needs to be modified to use MPS instead of CUDA. Here are the key files that need to be modified:

#### a. Modify inference.py

Open the `inference.py` file and modify the device setup section:

```bash
# Open the file in your preferred text editor
nano inference.py
```

Find the device setup section (around line 81) that looks like:

```python
# --- Device Setup ---
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
```

Replace it with:

```python
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
```

#### b. Modify train.py

Open the `train.py` file and make similar changes:

```bash
# Open the file in your preferred text editor
nano train.py
```

Find the device setup section (around line 270) and replace it with the same code as above.

Also, modify the `p_sample_loop` function to add MPS synchronization:

```python
@torch.no_grad()
def p_sample_loop(model: nn.Module, shape: tuple, timesteps: int,
                  betas: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor,
                  sqrt_recip_alphas: torch.Tensor, posterior_variance: torch.Tensor,
                  device: torch.device, image_input: torch.Tensor) -> torch.Tensor:
    """
    Performs the full DDPM sampling loop, starting from noise.
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

    # img now holds the predicted x_0

    # Final synchronization for MPS device
    if device.type == "mps":
        torch.mps.synchronize()
        # Optional: clear cache to free up memory
        torch.mps.empty_cache()

    return img
```

Also update the DataLoader configurations to support MPS:

```python
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True if device.type in ['cuda', 'mps'] else False,
                              collate_fn=custom_collate_fn)
```

And similarly for the evaluation DataLoader:

```python
eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True if device.type in ['cuda', 'mps'] else False,
                             collate_fn=custom_collate_fn)
```

## Running Inference

To run inference with the Diffusion Policy model using MPS acceleration:

```bash
python inference.py --data_dir ../mycobot_episodes/ --checkpoint_path ./checkpoints/model_best.pth --visualize_trajectory
```

You can also specify additional parameters:

```bash
python inference.py --data_dir ../mycobot_episodes/ --checkpoint_path ./checkpoints/model_best.pth --episode_id 1 --visualize_trajectory
```

## Training the Model

To train the model using MPS acceleration:

```bash
python train.py --data_dir ../mycobot_episodes/ --output_dir ./checkpoints --num_epochs 801 --batch_size 64 --save_interval 50 --eval_interval 50
```

You may need to adjust the batch size depending on your Mac's memory. If you encounter out-of-memory errors, try reducing the batch size:

```bash
python train.py --data_dir ../mycobot_episodes/ --output_dir ./checkpoints --num_epochs 801 --batch_size 32 --save_interval 50 --eval_interval 50
```

## Resuming Training

To resume training from a checkpoint:

```bash
python train.py --data_dir ../mycobot_episodes/ --output_dir ./checkpoints --num_epochs 801 --batch_size 64 --save_interval 50 --eval_interval 50 --resume_from ./checkpoints/model_best.pth
```

## Troubleshooting

### Virtual Environment Issues

If you encounter issues with the virtual environment:

1. **Missing activate script**: If `dp_venv/bin/activate` is missing:
   ```bash
   # Remove the corrupted environment
   rm -rf dp_venv

   # Create a new environment
   python3 -m venv dp_venv

   # Verify the structure
   ls -la dp_venv/bin
   ```

2. **Permission issues**: If you get permission errors:
   ```bash
   chmod -R u+w dp_venv
   ```

3. **Alternative using conda**: If venv continues to cause problems:
   ```bash
   # Install Miniconda (for Apple Silicon)
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   sh Miniconda3-latest-MacOSX-arm64.sh

   # Create and activate environment
   conda create -n diffusion_policy python=3.9
   conda activate diffusion_policy
   ```

### MPS Availability Issues

If MPS is not available:

1. **Check macOS version**: Ensure you're running macOS 12.3 or later:
   ```bash
   sw_vers -productVersion
   ```

2. **Check PyTorch version**: Ensure you have a version with MPS support:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

   PyTorch 1.12 or later is required for MPS support.

3. **Reinstall PyTorch**: If needed, reinstall with the correct version:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio
   ```

### Memory Issues

If you encounter memory errors when using MPS:

1. **Reduce batch size**: Try halving your batch size.

2. **Clear cache**: Add more cache clearing points in your code:
   ```python
   if torch.backends.mps.is_available():
       torch.mps.empty_cache()
   ```

3. **Monitor memory**: Use Activity Monitor to check memory usage.

4. **Restart Python**: Sometimes the MPS backend can accumulate memory that isn't properly released. Restart your Python process.

5. **Limit model size**: If possible, reduce model complexity or size.

### Performance Issues

If performance is slower than expected:

1. **Update PyTorch**: Make sure you're using the latest version of PyTorch.

2. **Optimize batch size**: Try different batch sizes to find the optimal one for your Mac.

3. **Check thermal throttling**: Use a cooling pad and ensure good ventilation.

4. **Close background apps**: Close other applications, especially those using GPU resources.

5. **Profile operations**: Some operations may not be optimized for MPS yet, so performance may vary.

### Operation Not Implemented Errors

If you see errors about operations not implemented for MPS:

1. **Update PyTorch**: Check if there's a newer PyTorch version that might support the operation.

2. **CPU fallback**: Implement a CPU fallback for the specific operation:
   ```python
   # Example of CPU fallback for a specific operation
   if device.type == "mps":
       # Move tensors to CPU, perform operation, then move back
       result = some_operation(tensor.cpu()).to(device)
   else:
       result = some_operation(tensor)
   ```

3. **Report the issue**: Report the issue to the [PyTorch GitHub repository](https://github.com/pytorch/pytorch/issues).

4. **Check workarounds**: Search the PyTorch forums for workarounds for specific operations.

### Training Crashes or Hangs

If training crashes or hangs:

1. **Reduce complexity**: Start with a simpler model or smaller dataset.

2. **Debug with CPU**: Try running on CPU first to isolate MPS-specific issues:
   ```bash
   PYTORCH_DISABLE_MPS=1 python train.py --batch_size 8
   ```

3. **Add logging**: Add more logging to identify where the issue occurs.

4. **Gradual scaling**: Start with a very small batch size and gradually increase.

5. **Check for NaN values**: Add checks for NaN values in your training loop:
   ```python
   if torch.isnan(loss):
       print("NaN loss detected!")
       # Handle accordingly
   ```

## Training with MPS Acceleration

### Starting Training

After setting up the environment and modifying the code, you can start training with MPS acceleration using the following command:

```bash
python train.py \
  --data_dir ../mycobot_episodes/ \
  --output_dir ./checkpoints \
  --num_epochs 200 \
  --batch_size 32 \
  --save_interval 10 \
  --eval_interval 10 \
  --learning_rate 1e-4 \
  --num_workers 2 \
  --eval_batch_size 16
```

### Recommended Parameters for Mac with MPS

- `--batch_size`: Start with 32 and adjust based on your Mac's memory. High-end Macs (M1 Max/Pro, M2/M3) might handle 64.
- `--num_epochs`: 200-800 depending on your dataset and patience.
- `--save_interval`: 10 is a good balance between checkpoint frequency and disk usage.
- `--eval_interval`: 10 allows regular evaluation without slowing down training too much.
- `--num_workers`: 2-4 is usually optimal for Macs.
- `--eval_batch_size`: Keep this smaller than training batch size (16 is a good starting point).

### Verifying MPS Usage During Training

When training starts, check the log output for:
```
Using MPS device (Apple Metal)
Device: mps
```

This confirms that the script is using the MPS device.

### Optimizing Training Performance

1. **Adjust Batch Size**: If you encounter out-of-memory errors, reduce the batch size. If training is stable but slow, try increasing it.

2. **Clear MPS Cache Periodically**: The code modifications already include cache clearing, but if you notice memory buildup, you can add more cache clearing points.

3. **Close Other Applications**: Close memory-intensive applications while training.

4. **Monitor Temperature**: Macs can throttle performance if they get too hot. Ensure good ventilation.

5. **Use a Cooling Pad**: For extended training sessions, consider using a cooling pad to prevent thermal throttling.

## Monitoring MPS Usage

### Using Activity Monitor

To monitor GPU usage on your Mac while running the model:

1. Open Activity Monitor (Applications > Utilities > Activity Monitor)
2. Go to the "GPU" tab
3. Look for the Python process running your model
4. Check the "GPU Usage" column - it should show significant usage (often 30-90%)

### Using Terminal Commands

You can also monitor GPU usage from the terminal:

```bash
# Check overall GPU usage
sudo powermetrics --samplers gpu_power -i 1000 | grep "GPU Active Utilization"

# Monitor memory pressure
vm_stat 1
```

### Checking Training Progress

Monitor the training progress by watching the loss values in the terminal output. You should see the loss decreasing over time.

You can also use TensorBoard to visualize training metrics:

```bash
# Install TensorBoard if not already installed
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=./checkpoints

# Open the provided URL in your browser (usually http://localhost:6006)
```

## Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Developer: Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Apple Developer: Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [PyTorch Forums - MPS Discussion](https://discuss.pytorch.org/c/mac/20)
