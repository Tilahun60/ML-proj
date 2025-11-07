# Network Log Anomaly Detection - LOCAL VERSION

This folder contains the local version of the project for running on your own machine.

## Quick Start

### 1. Setup

```bash
cd local
pip install -r ../requirements.txt
```

### 2. Prepare Dataset

Place CICIDS2017 CSV files in the `data/` directory:
- Recommended: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Or use any other CSV file from the dataset

### 3. Run

```bash
python main.py
```

## Configuration

Edit `config.py` to adjust:
- **SELECTED_FILES**: Which CSV file(s) to use
- **SEQUENCE_LENGTH**: Time window size (default: 50)
- **MAX_SAMPLES**: Limit dataset size for memory (default: 500000)
- **TRAINING_CONFIG**: Batch size, epochs, etc.

## Expected Time

- **With GPU**: ~2-3 hours
- **Without GPU (CPU)**: ~4-6 hours

## Memory Requirements

- **Single file (DDoS)**: ~6.5 GB RAM
- **All files**: ~55 GB RAM

## Project Structure

```
local/
├── main.py              # Main execution script
├── config.py           # Configuration file
├── requirements.txt    # Python dependencies (link to parent)
├── src/                # Source code modules
│   ├── data_preprocessing.py
│   ├── transformer_model.py
│   ├── lstm_model.py
│   ├── random_forest_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── data/               # Place CSV files here
├── models/             # Trained models saved here
└── results/            # Results and visualizations
```

## Troubleshooting

### Out of Memory
- Reduce `MAX_SAMPLES` in `config.py`
- Reduce `SEQUENCE_LENGTH` to 30
- Use single file instead of all files

### Slow Training
- Enable GPU if available
- Reduce `num_epochs` in `TRAINING_CONFIG`
- Increase `batch_size` if you have more RAM

## See Also

- `../colab/` - Colab version (faster with GPU)
- `../README.md` - Main project documentation

