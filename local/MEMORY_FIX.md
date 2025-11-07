# Memory Issue Fix

## Problem
The CICIDS2017 dataset has **2.8 million samples**. When creating time-series sequences:
- With `sequence_length=100`: Requires ~115 GB RAM
- With `sequence_length=50`: Requires ~55 GB RAM

## Solutions Applied

1. **Reduced Sequence Length**: Changed from 100 to 50 in `config.py`
   - This halves the memory requirement
   - Still captures temporal patterns effectively

2. **Added Sampling Option**: Added `MAX_SAMPLES` parameter in `config.py`
   - Default: 500,000 samples (manageable for most systems)
   - Set to `None` to use full dataset (requires ~55GB RAM with seq_len=50)
   - Set to a smaller number (e.g., 100000) for limited memory systems

3. **Fixed Infinity Values**: Added robust handling for infinity/NaN values
   - Detects and replaces infinity values
   - Clips extreme values to reasonable ranges
   - Prevents errors during normalization

## Memory Requirements

| Configuration | RAM Needed |
|--------------|------------|
| Full dataset (2.8M), seq_len=100 | ~115 GB |
| Full dataset (2.8M), seq_len=50 | ~55 GB |
| 500K samples, seq_len=50 | ~10 GB |
| 100K samples, seq_len=50 | ~2 GB |

## Recommendations

1. **For systems with < 16GB RAM**: 
   - Set `MAX_SAMPLES = 100000` in `config.py`
   - This provides enough data for training while staying within memory limits

2. **For systems with 16-32GB RAM**:
   - Set `MAX_SAMPLES = 500000` (current default)
   - Good balance between data size and memory usage

3. **For systems with > 32GB RAM**:
   - Set `MAX_SAMPLES = None` to use full dataset
   - Or increase to 1,000,000+ samples

4. **For GPU training**:
   - Can use larger datasets as GPU memory is separate
   - Still limited by CPU RAM for data loading

## How to Adjust

Edit `config.py`:
```python
SEQUENCE_LENGTH = 50  # Reduce further if needed (e.g., 30)
MAX_SAMPLES = 100000  # Adjust based on your RAM
```

The sampling is done **after** combining all CSV files but **before** creating sequences, so it maintains the class distribution.

