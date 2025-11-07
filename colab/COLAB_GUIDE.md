# Running on Google Colab - Complete Guide

## Time Estimates

### With GPU (T4/V100 - Free/Pro):
- **Data Preprocessing**: 5-10 minutes
- **Transformer Training**: 30-60 minutes (30 epochs with early stopping)
- **LSTM Training**: 20-40 minutes (30 epochs with early stopping)  
- **Random Forest**: 5-10 minutes
- **Evaluation & Visualization**: 2-5 minutes
- **TOTAL: ~1.5-2 hours**

### Without GPU (CPU only):
- **TOTAL: ~4-6 hours**

## Quick Start Steps

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Enable GPU
- Click: **Runtime → Change runtime type → GPU (T4 or V100)**
- This is **essential** for reasonable training times

### 3. Upload Project Files

**Option A: Upload via Colab UI**
1. Click the folder icon (left sidebar)
2. Click "Upload" and select all project files:
   - `config.py`
   - `main.py`
   - `requirements.txt`
   - All files from `src/` directory

**Option B: Use Google Drive (Recommended for large datasets)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy project files from Drive
!cp -r /content/drive/MyDrive/ML-proj/* /content/
```

### 4. Upload Dataset
- Upload the CICIDS2017 CSV file(s) to Colab
- Recommended: Just upload `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Place in `/content/data/` directory

### 5. Install Dependencies
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

### 6. Update Config for Colab
Edit `config.py` or create a Colab-specific config:

```python
# Colab-optimized settings
SELECTED_FILES = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
SEQUENCE_LENGTH = 50
MAX_SAMPLES = None  # Use full file (225K is fine)

TRAINING_CONFIG = {
    "batch_size": 128,  # Larger for GPU
    "learning_rate": 0.001,
    "num_epochs": 30,  # Reduced for faster training
    "early_stopping_patience": 5,
    "device": "cuda"
}
```

### 7. Run Training
```python
import os
os.chdir('/content')  # Change to project directory

# Run main script
!python main.py
```

## Optimizations for Faster Training

### Speed Up Training:
1. **Reduce epochs**: `num_epochs = 20` (instead of 50)
2. **Increase batch size**: `batch_size = 256` (if GPU memory allows)
3. **Smaller model**: Reduce `d_model = 64` (instead of 128)
3. **Sample data**: `MAX_SAMPLES = 100000` (use subset)
4. **Reduce sequence length**: `SEQUENCE_LENGTH = 30` (instead of 50)

### Memory Optimizations:
1. Use single file (already configured)
2. Set `MAX_SAMPLES = 100000` if you get OOM errors
3. Reduce batch size if needed: `batch_size = 64`

## Colab Limitations

### Free Tier:
- **GPU**: T4 (16GB) - Limited to ~12 hours continuous
- **RAM**: ~12-15GB
- **Session timeout**: After 90 minutes of inactivity

### Pro Tier ($10/month):
- **GPU**: T4 or V100 (better performance)
- **RAM**: ~32GB
- **Longer sessions**: Up to 24 hours

## Monitoring Progress

Add progress tracking in Colab:

```python
# In main.py or notebook, add:
from tqdm import tqdm
import time

start_time = time.time()
# ... training code ...
elapsed = time.time() - start_time
print(f"Training completed in {elapsed/60:.1f} minutes")
```

## Downloading Results

After training completes:

```python
from google.colab import files
import os

# Download visualizations
for file in os.listdir('results'):
    if file.endswith('.png'):
        files.download(f'results/{file}')

# Download models
for file in os.listdir('models'):
    if file.endswith('.pth'):
        files.download(f'models/{file}')
```

## Troubleshooting

### Out of Memory (OOM):
- Reduce `batch_size` to 64 or 32
- Set `MAX_SAMPLES = 100000`
- Reduce `SEQUENCE_LENGTH` to 30

### Session Disconnected:
- Colab disconnects after 90 min inactivity
- Add periodic print statements to keep session alive
- Use Pro tier for longer sessions

### Slow Training:
- Ensure GPU is enabled (Runtime → Change runtime type)
- Check GPU usage: `!nvidia-smi`
- Increase batch size if memory allows

## Expected Output Times

With **T4 GPU** and **single DDoS file** (225K samples):

| Stage | Time (GPU) | Time (CPU) |
|-------|-----------|------------|
| Data Loading | 1-2 min | 2-3 min |
| Preprocessing | 3-5 min | 5-8 min |
| Sequence Creation | 2-3 min | 3-5 min |
| Transformer Training | 30-60 min | 2-3 hours |
| LSTM Training | 20-40 min | 1.5-2 hours |
| Random Forest | 5-10 min | 10-15 min |
| Evaluation | 2-3 min | 5-8 min |
| **TOTAL** | **1.5-2 hours** | **4-6 hours** |

## Tips

1. **Save checkpoints**: Colab can disconnect, save models periodically
2. **Use Drive**: Save important files to Google Drive
3. **Monitor GPU**: `!nvidia-smi` to check GPU utilization
4. **Early stopping**: Already configured, will stop if no improvement
5. **Reduce epochs**: 20-30 epochs often sufficient with early stopping

## Quick Colab Notebook

See `Network_Anomaly_Detection_Colab.ipynb` for a ready-to-use Colab notebook with all steps included.

