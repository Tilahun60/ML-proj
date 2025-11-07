"""
Quick start script for Google Colab
Run this in a Colab cell after uploading project files
"""

# Step 1: Install dependencies (run in separate cell first)
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy

# Step 2: Setup paths
import os
import sys

# Adjust paths for Colab
if '/content' in os.getcwd() or 'google.colab' in str(sys.modules):
    print("Running in Google Colab - applying Colab optimizations...")
    
    # Update config for Colab
    import config
    
    # Colab-optimized training config
    config.TRAINING_CONFIG = {
        "batch_size": 128,  # Larger for GPU
        "learning_rate": 0.001,
        "num_epochs": 30,  # Reduced for faster training
        "early_stopping_patience": 5,  # Faster early stopping
        "device": "cuda"
    }
    
    # Use single best file
    config.SELECTED_FILES = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    config.SEQUENCE_LENGTH = 50
    config.MAX_SAMPLES = None  # Use full file (225K is fine for Colab)
    
    print("Colab optimizations applied!")
    print(f"Batch size: {config.TRAINING_CONFIG['batch_size']}")
    print(f"Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"Selected file: {config.SELECTED_FILES}")

# Step 3: Check GPU
import torch
if torch.cuda.is_available():
    print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n⚠️  No GPU detected - training will be slower (4-6 hours vs 1.5-2 hours)")

# Step 4: Run main script
print("\n" + "="*80)
print("Starting training...")
print("="*80)

from main import main
main()

print("\n" + "="*80)
print("Training completed!")
print("="*80)
print("\nTo download results, run:")
print("  from google.colab import files")
print("  files.download('results/model_comparison.png')")

