# Network Log Anomaly Detection using Transformer on Time-Series Data

A deep learning project that implements and compares Transformer, LSTM, and Random Forest models for detecting anomalies in network traffic logs.

## Project Structure

This project is organized into two main folders:

```
ML-proj/
├── local/          # For local execution in Cursor/VS Code
└── colab/          # For Google Colab execution
```

## Quick Start

### 🖥️ Local Execution (Cursor/VS Code)

```bash
cd local
pip install -r requirements.txt
# Place CICIDS2017 CSV files in local/data/
python main.py
```

See `local/README.md` for detailed instructions.

### ☁️ Google Colab Execution

1. Open Google Colab: https://colab.research.google.com/
2. Enable GPU: **Runtime → Change runtime type → GPU**
3. Upload files from `colab/` folder
4. Run `Network_Anomaly_Detection_Colab.ipynb`

See `colab/README.md` for detailed instructions.

## Features

- **Transformer-based anomaly detection** with attention mechanisms
- **LSTM baseline** for sequential pattern recognition
- **Random Forest baseline** for traditional ML comparison
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Visualization of training curves, confusion matrices, and model comparisons
- Support for imbalanced datasets with class weighting

## Dataset

This project uses the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity.

**Dataset Source:** https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains:
- Normal (BENIGN) and attack traffic
- Attack types: DoS, PortScan, DDoS, Web attacks, Infiltration
- Flow-based features (duration, bytes, packets, flags)
- Time-based metrics

## Expected Performance

| Environment | Time | Notes |
|-------------|------|-------|
| Local CPU | 4-6 hours | Slow but free |
| Local GPU | 2-3 hours | Fast if you have GPU |
| Colab GPU (Free) | 1.5-2 hours | **Best option for most users** |
| Colab CPU | 4-6 hours | Not recommended |

## Model Architectures

### Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for temporal information
- Encoder layers with feedforward networks
- Global average pooling for sequence classification

### LSTM Model
- Bidirectional LSTM layers
- Captures forward and backward temporal dependencies
- Fully connected classification head

### Random Forest Model
- Traditional ensemble method
- Flattens time-series sequences
- Balanced class weights for imbalanced data

## Evaluation Metrics

The project evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

See `local/requirements.txt` or `colab/requirements.txt` for complete list.

## References

1. Vaswani et al., *"Attention is All You Need,"* NeurIPS 2017
2. Canadian Institute for Cybersecurity, *CICIDS2017 Dataset*
3. Hariri et al., *"Efficient Anomaly Detection in Network Traffic using Transformers,"* IEEE Access, 2022

## License

This project is for educational and research purposes.

## Getting Help

- **Local setup issues**: See `local/README.md`
- **Colab setup issues**: See `colab/README.md` or `colab/COLAB_GUIDE.md`
- **Dataset questions**: See `local/analyze_files.py` to inspect your dataset
