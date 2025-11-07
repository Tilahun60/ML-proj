# Project Structure

The project is now organized into two main folders for different execution environments:

## 📁 Folder Organization

```
ML-proj/
├── local/                    # Local execution version
│   ├── main.py              # Main script for local
│   ├── config.py            # Local configuration
│   ├── requirements.txt     # Dependencies
│   ├── README.md            # Local setup guide
│   ├── src/                 # Source code modules
│   │   ├── data_preprocessing.py
│   │   ├── transformer_model.py
│   │   ├── lstm_model.py
│   │   ├── random_forest_model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── visualize.py
│   ├── data/                # Place CSV files here
│   ├── models/              # Trained models
│   └── results/             # Results and visualizations
│
├── colab/                    # Google Colab version
│   ├── main.py              # Main script for Colab
│   ├── config.py            # Colab-optimized configuration
│   ├── requirements.txt     # Dependencies
│   ├── README.md            # Colab setup guide
│   ├── src/                 # Source code modules (same as local)
│   ├── data/                # Upload CSV files here in Colab
│   ├── models/              # Trained models
│   └── results/             # Results and visualizations
│
├── data/                     # Original data location (reference)
├── Network_Anomaly_Detection_Colab.ipynb  # Ready-to-use Colab notebook
├── README.md                # Main project documentation
├── COLAB_GUIDE.md           # Detailed Colab guide
└── [other documentation files]
```

## 🚀 Quick Start

### Local Version
```bash
cd local
pip install -r requirements.txt
# Place CSV files in local/data/
python main.py
```

### Colab Version
1. Upload `colab/` folder contents to Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run the cells in `Network_Anomaly_Detection_Colab.ipynb`

## 🔑 Key Differences

| Feature | Local | Colab |
|---------|-------|-------|
| **Batch Size** | 64 | 128 |
| **Epochs** | 50 | 30 |
| **Early Stopping** | 10 | 5 |
| **Paths** | Relative | Absolute (/content/ML-proj/) |
| **GPU Check** | Basic | Detailed with info |
| **Time Tracking** | No | Yes |
| **Expected Time** | 2-6 hours | 1.5-2 hours (GPU) |

## 📝 Configuration Files

Both versions have their own `config.py`:
- **local/config.py**: Optimized for local CPU/GPU
- **colab/config.py**: Optimized for Colab GPU with faster settings

## 📦 Shared Code

The `src/` modules are identical in both folders - they contain the core functionality:
- Data preprocessing
- Model architectures
- Training utilities
- Evaluation and visualization

## 🎯 Which Version to Use?

### Use **Local** if:
- You have a local GPU setup
- You want full control over the environment
- You're doing development/debugging
- You have sufficient RAM (8GB+)

### Use **Colab** if:
- You don't have a local GPU
- You want faster training (free GPU)
- You're doing a one-time run
- You want to share results easily

## 📊 Performance Comparison

| Environment | Time | Notes |
|-------------|------|-------|
| Local CPU | 4-6 hours | Slow but free |
| Local GPU | 2-3 hours | Fast if you have GPU |
| Colab GPU (Free) | 1.5-2 hours | Best option for most users |
| Colab CPU | 4-6 hours | Not recommended |

## 🔄 Migration

To switch between versions:
1. Copy your data files to the appropriate `data/` folder
2. Use the corresponding `main.py` and `config.py`
3. All source code (`src/`) is identical

