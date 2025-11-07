# Network Log Anomaly Detection - COLAB VERSION

This folder contains the Google Colab optimized version of the project.

## Quick Start

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Enable GPU
- **Runtime → Change runtime type → GPU (T4 or V100)**
- This is essential for reasonable training times!

### 3. Upload Files
Upload all files from this `colab/` folder:
- `main.py`
- `config.py`
- All files from `src/` directory
- CSV dataset file: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### 4. Run in Colab Notebook

```python
# Install dependencies
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
%pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy

# Setup paths
import os
os.makedirs('/content/ML-proj/data', exist_ok=True)
os.makedirs('/content/ML-proj/models', exist_ok=True)
os.makedirs('/content/ML-proj/results', exist_ok=True)
os.makedirs('/content/ML-proj/src', exist_ok=True)

# Upload files using Colab UI or Drive, then:
import sys
sys.path.append('/content/ML-proj')

# Update config paths
import config
config.DATA_DIR = '/content/ML-proj/data'
config.MODEL_DIR = '/content/ML-proj/models'
config.RESULTS_DIR = '/content/ML-proj/results'

# Run training
from main import main
main()
```

## Expected Time

- **With GPU (T4/V100)**: ~1.5-2 hours total
- **Without GPU (CPU)**: ~4-6 hours total

### Breakdown:
- Data Preprocessing: 5-10 minutes
- Transformer Training: 30-60 minutes
- LSTM Training: 20-40 minutes
- Random Forest: 5-10 minutes
- Evaluation: 2-5 minutes

## Colab Optimizations

This version includes:
- **Larger batch size**: 128 (vs 64 local) for better GPU utilization
- **Reduced epochs**: 30 (vs 50 local) with faster early stopping
- **GPU-optimized**: Automatically uses CUDA if available

## Configuration

The `config.py` is pre-configured for Colab:
- Uses single DDoS file (225K samples)
- Batch size: 128
- Epochs: 30
- Early stopping patience: 5

## Download Results

After training:

```python
from google.colab import files
import os

# Download visualizations
for file in os.listdir('/content/ML-proj/results'):
    if file.endswith('.png'):
        files.download(f'/content/ML-proj/results/{file}')
```

## Using Colab Notebook

See `../Network_Anomaly_Detection_Colab.ipynb` for a ready-to-use notebook with all steps.

## Troubleshooting

### Out of Memory
- Reduce `MAX_SAMPLES` in `config.py`
- Reduce `batch_size` to 64
- Use smaller model: reduce `d_model` to 64

### Session Disconnected
- Colab disconnects after 90 min inactivity
- Use Pro tier for longer sessions
- Add periodic prints to keep session alive

### Slow Training
- Ensure GPU is enabled
- Check GPU usage: `!nvidia-smi`
- Increase batch size if memory allows

## See Also

- `../local/` - Local version
- `../COLAB_GUIDE.md` - Detailed Colab guide
- `../README.md` - Main project documentation

